"""Forecast credited team units and allocate them across each full match sheet."""

from collections.abc import Mapping
import warnings

import numpy as np
import pandas as pd

from six_nations_fantasy.data.lineup import match_context, substitute_status
from six_nations_fantasy.score import POINTS_PER_UNIT

from ._share_regression import Allocation, TeamRate
from .position_shrinkage import _positions


POINTS_PER_STAT = {**POINTS_PER_UNIT, "carried_meters": 0.1}


def component_units(rows):
    """Read the counts modelled by the share regressions, including raw metres."""
    counts = rows[list(POINTS_PER_STAT)].copy()
    # The regression needs nonnegative counts. Treat a net loss of ground as
    # zero metres for this model. The evaluation still uses simple_score().
    counts["carried_meters"] = counts["carried_meters"].clip(lower=0)
    values = rows[list(POINTS_PER_STAT)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values != np.floor(values)).any():
        raise ValueError("Expected complete, integer scoring stats")
    if (counts.to_numpy(dtype=float) < 0).any():
        raise ValueError("Scoring events must be nonnegative")
    return counts


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

    # The team table has one row per match side. A player's team_rows entry
    # is the integer index of their side in that table. For example, if rows
    # 0 and 1 are France and Ireland, team_rows=[0, 1, 0] maps two players
    # to France and one to Ireland, even if the player rows are interleaved.
    team_rows, sides = pd.factorize(pd.MultiIndex.from_frame(rows[keys]), sort=False)
    columns = ["match_id", "date", "team", "opponent", "competition", "home"]
    teams = rows.drop_duplicates(keys)[columns].set_index(keys).loc[sides].reset_index()
    return rows, team_rows, teams


class PlayerShareModel:
    """Predict each team's scoring stats and divide them among its 23 players."""

    def __init__(self, tau=0.25, *, tau_by_stat: Mapping[str, float] | None = None):
        """Set opponent shrinkage for all stats, with optional per-stat overrides."""
        overrides = dict(tau_by_stat or {})
        unknown = overrides.keys() - POINTS_PER_STAT.keys()
        if unknown:
            raise ValueError(f"Unknown scoring stats: {sorted(unknown)}")
        if any(not np.isfinite(scale) or scale < 0 for scale in (tau, *overrides.values())):
            raise ValueError("Opponent scales must be finite and nonnegative")
        self.opponent_scales = {stat: overrides.get(stat, tau) for stat in POINTS_PER_STAT}

    def fit(self, history):
        frame = match_context(history.performances, history.matches)
        frame = frame.assign(date=frame["match_id"].map(history.matches.set_index("match_id")["date"]))
        rows, team_rows, teams = full_lineups(frame, training=True)
        units = component_units(rows)
        models = {}
        for stat, tau in self.opponent_scales.items():
            counts = units[stat].to_numpy(dtype=float)
            # Sum the player counts into one observation per team row.
            totals = np.bincount(team_rows, weights=counts)
            try:
                rate = TeamRate(teams, totals, tau)
                allocation = Allocation(rows, team_rows, totals, counts)
            except RuntimeError as exc:
                raise RuntimeError(f"Share model fit failed for {stat} (tau={tau:g}): {exc}") from exc
            models[stat] = rate, allocation
        self.models_ = models

    def _prediction_lineups(self, fixtures):
        if not hasattr(self, "models_"):
            raise RuntimeError("Call fit() before predict()")
        return full_lineups(fixtures, training=False)

    def predict_components(self, fixtures, *, observed_team_units=None):
        """Return predicted counts per player, with optional actual totals for diagnostics."""
        rows, team_rows, teams = self._prediction_lineups(fixtures)
        if observed_team_units is not None:
            observed = observed_team_units.reindex(pd.MultiIndex.from_frame(teams[["match_id", "team"]]))
            values = observed[list(self.models_)].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Missing or invalid observed team totals")
        predicted = {}
        for stat, (rate, allocation) in self.models_.items():
            totals = rate.predict(teams) if observed_team_units is None else observed[stat].to_numpy()
            predicted[stat] = totals[team_rows] * allocation.predict(rows, team_rows)
        return pd.DataFrame(predicted, index=rows.index)

    def explain_component(self, fixtures, stat):
        """Return the team forecast and each player's share for one stat."""
        rows, team_rows, teams = self._prediction_lineups(fixtures)
        rate, allocation = self.models_[stat]
        factors = rate.multipliers(teams)
        if "opponent_multiplier" not in factors:
            factors["opponent_multiplier"] = 1.0
        team_forecast = teams.join(factors).assign(
            baseline_units=rate.base_rate, expected_units=rate.predict(teams)
        )
        player_forecast = rows[["match_id", "team", "player_id", "number", "position_role"]].copy()
        weights = allocation.multipliers(rows)
        for column in weights:
            player_forecast[column] = weights[column].to_numpy()
        player_forecast["share"] = allocation.predict(rows, team_rows)
        player_forecast["expected_units"] = (
            team_forecast["expected_units"].to_numpy()[team_rows] * player_forecast["share"]
        )
        return team_forecast, player_forecast

    def predict(self, fixtures):
        units = self.predict_components(fixtures)
        return units.mul(pd.Series(POINTS_PER_STAT)).sum(axis=1).rename("predicted")
