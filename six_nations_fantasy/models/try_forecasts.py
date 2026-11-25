"""Compare three ways to divide a team's expected tries among its 23 players."""

import numpy as np
import pandas as pd

from six_nations_fantasy.data.lineup import match_context

from six_nations_fantasy.score import POINTS_PER_UNIT, simple_score

from .recency_weighted_mean import RecencyWeightedMean
from ._mean import MeanScores
from ._share_regression import TeamRate
from .player_share import full_lineups, lineup_share


class TryForecasts:
    def __init__(self, *, prior_opportunities=50, half_life_days=None):
        """The adjusted estimate gives the shirt average this many team tries of weight."""
        if not np.isfinite(prior_opportunities) or prior_opportunities <= 0:
            raise ValueError("prior_opportunities must be positive and finite")
        if half_life_days is not None and (not np.isfinite(half_life_days) or half_life_days <= 0):
            raise ValueError("half_life_days must be positive and finite")
        self.prior_opportunities = prior_opportunities
        self.half_life_days = half_life_days

    def fit(self, history):
        performances = match_context(history.player_matches(), history.matches)
        rows, sides, teams = full_lineups(performances, training=True)
        tries = rows["tries"]
        minutes = rows["minutes_played"]
        if not minutes.between(0, 80).all() or not (tries.ge(0) & tries.mod(1).eq(0)).all():
            raise ValueError("Expected minutes from 0 to 80 and nonnegative whole try counts")
        team_tries = np.bincount(sides, weights=tries)
        self.team_rate = TeamRate(teams, team_tries, 0.25)
        age = (rows["date"].max() - rows["date"]).dt.days
        self.recency = MeanScores(rows, tries, 0.5 ** (age / 180))
        weight = 1.0 if self.half_life_days is None else 0.5 ** (age / self.half_life_days)

        # Try times are unavailable. Approximate how many team tries occurred
        # during each appearance using the fraction of the match played.
        rows["opportunities"] = team_tries[sides] * minutes / 80 * weight
        rows["weighted_tries"] = tries * weight
        rows["weighted_minutes"] = minutes * weight
        rows["weight"] = weight
        rows["shirt"] = rows["number"].astype(int)
        shirts = rows.groupby("shirt")[["weighted_tries", "opportunities", "weighted_minutes", "weight"]].sum()
        # One average opportunity keeps rarely observed shirts above zero.
        average = rows["weighted_tries"].sum() / max(rows["opportunities"].sum(), 1)
        self.shirt_rate = (shirts["weighted_tries"] + average) / (shirts["opportunities"] + 1)
        self.shirt_minutes = shirts["weighted_minutes"] / shirts["weight"]
        self.by_player = rows.groupby("player_id")[["weighted_tries", "opportunities"]].sum()

    def predict_shares(self, fixtures):
        rows, sides, teams = full_lineups(fixtures, training=False)
        shirt = rows["number"].astype(int)
        rate = shirt.map(self.shirt_rate)
        minutes = shirt.map(self.shirt_minutes)
        players = rows["player_id"]
        tries = players.map(self.by_player["weighted_tries"]).fillna(0)
        opportunities = players.map(self.by_player["opportunities"]).fillna(0)
        k = self.prior_opportunities
        adjusted_rate = (tries + k * rate) / (opportunities + k)
        shares = pd.DataFrame({
            "Shirt only": lineup_share(rate * minutes, sides),
            "Recency": lineup_share(self.recency.predict(rows), sides),
            "Adjusted player": lineup_share(adjusted_rate * minutes, sides),
        })
        totals = pd.Series(self.team_rate.predict(teams)[sides], index=rows.index)
        return shares, totals


class RecencyWithAdjustedTries:
    """Use the ordinary 180-day recency mean for every point except tries."""

    def __init__(self):
        self.other_points = RecencyWeightedMean(scorer=lambda rows: simple_score(rows, try_points=0))
        self.tries = TryForecasts()

    def fit(self, history):
        self.other_points.fit(history)
        self.tries.fit(history)

    def predict(self, fixtures):
        shares, team_tries = self.tries.predict_shares(fixtures)
        expected_tries = shares["Adjusted player"] * team_tries
        return self.other_points.predict(fixtures) + POINTS_PER_UNIT["tries"] * expected_tries
