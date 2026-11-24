"""Forecast credited team units and allocate them across each full match sheet."""

import warnings

import numpy as np
import pandas as pd

from six_nations_fantasy.data.lineup import match_context, substitute_status
from six_nations_fantasy.score import POINTS_PER_UNIT, scoring_units

from ._share_regression import Allocation, TeamRate, choose_tau
from .position_shrinkage import _positions


def component_units(rows):
    units = pd.DataFrame({stat: scoring_units(rows, stat) for stat in POINTS_PER_UNIT})

    #TODO I don't think this is necessary. I don't believe that there are negative carried_meters entries
    #TODO I am happy to, in favour of simplicity, forget the metres//10 logic and just use 0.1 multiplier, since
    # this will not make the difference in prediction

    # A player can lose ground. Preserve floor(metres/10) exactly with two counts.
    metres = units["carried_meters"]
    units["lost_metre_units"] = (-metres).clip(lower=0)
    units["carried_meters"] = metres.clip(lower=0)
    values = units.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Expected complete, nonnegative integer scoring units")
    return units


def full_lineups(frame, *, training):
    keys = ["match_id", "team"]
    required = [*keys, "date", "player_id", "number", "competition", "opponent", "home"]
    missing = set(required) - set(frame)
    if missing or frame[required].isna().any().any():
        raise ValueError(f"Missing fixture context or lineup values; required: {required}")
    if frame.duplicated(["match_id", "player_id"]).any():
        raise ValueError("Duplicate player in a match lineup")
    shirts = pd.to_numeric(frame["number"], errors="raise")
    side_keys = [frame[col] for col in keys]
    grouped = shirts.groupby(side_keys, sort=False)
    valid_numbers = shirts.isin(range(1, 24)).groupby(side_keys).transform("all")
    complete = grouped.transform("size").eq(23) & grouped.transform("nunique").eq(23) & valid_numbers
    if not complete.all():
        incomplete = frame.loc[~complete, keys].drop_duplicates()
        message = f"{len(incomplete)} incomplete team sheets (expected shirts 1–23), first: {incomplete.iloc[0].to_dict()}"
        if not training:
            raise ValueError(message)
        warnings.warn("Excluded from training: " + message, stacklevel=2)
    rows = frame.loc[complete].copy()
    if rows.empty:
        raise ValueError("No complete team sheets")
    rows["position_role"] = _positions(rows, substitute_status(rows))

    #TODO I don't understand what is going on here and why
    # Group codes index one row per side in the team regression.
    groups = rows.groupby(keys, sort=False).ngroup().to_numpy()
    teams = rows.drop_duplicates(keys)[["match_id", "date", "team", "opponent", "competition", "home"]]
    return rows, groups, teams.reset_index(drop=True)


class PlayerShareModel:
    """Poisson team means times conditional multinomial player shares, by scoring unit."""

    def __init__(self, tau=0.25, *, component_tau=False):
        if not np.isfinite(tau) or tau < 0 or (component_tau and tau == 0):
            raise ValueError("tau must be finite and nonnegative; tuning requires positive tau")
        self.tau = tau
        self.component_tau = component_tau

    def fit(self, history):
        frame = match_context(history.performances, history.matches)
        frame = frame.assign(date=frame["match_id"].map(history.matches.set_index("match_id")["date"]))
        rows, groups, teams = full_lineups(frame, training=True)
        units = component_units(rows)
        dates = sorted(teams["date"].unique())

        #TODO what is this 0.8 magic number
        earlier = teams["date"].lt(dates[int(0.8 * len(dates))]).to_numpy()
        models, scales = {}, {}
        for stat in units:
            counts = units[stat].to_numpy(dtype=float)
            totals = np.bincount(groups, counts)
            tau = choose_tau(teams, totals, earlier, self.tau) if self.component_tau else self.tau
            models[stat] = (TeamRate(teams, totals, tau), Allocation(rows, groups, totals, counts))
            scales[stat] = tau
        self.models_ = models
        self.opponent_scales_ = pd.Series(scales, name="tau")

    def predict_components(self, fixtures, *, observed_team_units=None):
        """Return per-player expected units. Observed team totals are diagnostic only.

        Optional totals are indexed by (match_id, team) with component_units columns.
        """
        if not hasattr(self, "models_"):
            raise RuntimeError("Call fit() before predict()")
        rows, groups, teams = full_lineups(fixtures, training=False)
        if observed_team_units is not None:
            observed = observed_team_units.reindex(pd.MultiIndex.from_frame(teams[["match_id", "team"]]))
            values = observed[list(self.models_)].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Missing or invalid observed team totals")
        predicted = {}
        for stat, (rate, allocation) in self.models_.items():
            totals = rate.predict(teams) if observed_team_units is None else observed[stat].to_numpy()
            predicted[stat] = totals[groups] * allocation.predict(rows, groups)
        return pd.DataFrame(predicted, index=rows.index)

    def predict(self, fixtures):
        units = self.predict_components(fixtures)
        weights = {**POINTS_PER_UNIT, "lost_metre_units": -1}
        return units.mul(pd.Series(weights)).sum(axis=1).rename("predicted")
