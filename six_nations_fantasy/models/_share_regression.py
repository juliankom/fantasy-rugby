"""Fit the two regressions used by the player-share model."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack


class CategoricalDesign:
    """Remember which categories appeared in training and give each a coefficient.

    For example, team=A and opponent=A have separate columns. A fixture activates
    one column per field. An unknown opponent activates none of the opponent
    columns, so its multiplier is 1.
    """

    def __init__(self, frame, prior_scales):
        self.categories = {field: pd.Index(frame[field].unique()) for field in prior_scales}
        # A zero-centred normal prior with standard deviation s adds beta² / (2s²).
        self.precision = np.concatenate([
            np.full(len(values), 1 / prior_scales[field] ** 2)
            for field, values in self.categories.items()
        ])

    def matrix(self, frame):
        """Build the sparse matrix of 0s and 1s used in both fit and predict."""
        blocks = []
        for field, values in self.categories.items():
            codes = values.get_indexer(frame[field])
            rows = np.flatnonzero(codes >= 0)
            blocks.append(csr_matrix((np.ones(len(rows)), (rows, codes[rows])),
                                     shape=(len(frame), len(values))))
        return hstack(blocks, format="csr")

    def multipliers(self, frame, coefficients):
        """Show the factor contributed by each field before multiplying them."""
        factors = {}
        offset = 0
        for field, values in self.categories.items():
            fitted = coefficients[offset:offset + len(values)]
            factors[field] = np.exp(frame[field].map(pd.Series(fitted, index=values)).fillna(0))
            offset += len(values)
        return pd.DataFrame(factors, index=frame.index)


def minimize_loss(objective, size):
    """Fit ridge-penalized coefficients starting at zero (all multipliers equal 1)."""
    # L-BFGS-B uses the gradient returned by objective (jac=True). Stop when
    # the relative improvement in loss is below ftol, with a cap of 600 iterations.
    result = minimize(objective, np.zeros(size), jac=True, method="L-BFGS-B",
                      options={"maxiter": 600, "ftol": 1e-11})
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"Share model fit failed: {result.message}")
    return result.x


class TeamRate:
    """Predict one stat total for one side of a match using a Poisson regression."""

    def __init__(self, frame, counts, tau):
        self.base_rate = counts.mean()
        # These are hand-set standard deviations of log multipliers. Competition
        # gets more room to vary than team and home/away because its prior is wider.
        prior_scales = {"competition": 2.0, "home": 0.5, "team": 0.5}
        if tau > 0:
            prior_scales["opponent"] = tau
        self.design = CategoricalDesign(frame, prior_scales)
        design = self.design.matrix(frame)
        precision = self.design.precision

        def objective(coefficients):
            # Each row's log multiplier is the sum of its active coefficients.
            log_multiplier = design @ coefficients
            expected = self.base_rate * np.exp(log_multiplier)
            # For an observed count y and expected count mu, -log P(Y=y)
            # is mu - y*log(mu) + log(y!). Drop log(y!) and the fixed base-rate
            # part of log(mu), since neither changes when the coefficients move.
            loss = (expected - counts * log_multiplier).sum()
            loss += 0.5 * np.sum(precision * coefficients**2)
            gradient = design.T @ (expected - counts) + precision * coefficients
            return loss, gradient

        size = design.shape[1]
        self.coefficients = minimize_loss(objective, size) if self.base_rate else np.zeros(size)

    def predict(self, frame):
        return self.base_rate * np.exp(self.design.matrix(frame) @ self.coefficients)

    def multipliers(self, frame):
        factors = self.design.multipliers(frame, self.coefficients)
        return factors.rename(columns=lambda col: f"{col}_multiplier")


class Allocation:
    """Estimate relative player weights from how each team split its observed totals."""

    def __init__(self, rows, team_rows, totals, counts):
        # These scales allow shirt positions to differ more than two players in
        # the same position. Each player coefficient applies across positions.
        self.design = CategoricalDesign(rows, {"position_role": 2.0, "player_id": 0.75})
        design = self.design.matrix(rows)
        precision = self.design.precision

        def objective(coefficients):
            # The multinomial loss is -sum(y_i * log(share_i)), where y_i is
            # the player's count and shares sum to 1 within each team sheet.
            log_shares = self.log_shares(design @ coefficients, team_rows)
            loss = -np.sum(counts * log_shares) + 0.5 * np.sum(precision * coefficients**2)
            expected = totals[team_rows] * np.exp(log_shares)
            gradient = design.T @ (expected - counts) + precision * coefficients
            return loss, gradient

        size = design.shape[1]
        self.coefficients = minimize_loss(objective, size) if counts.any() else np.zeros(size)

    @staticmethod
    def log_shares(log_weights, team_rows):
        # Subtract the largest log weight on each sheet before exponentiating.
        # The same factor cancels from every player's numerator and denominator.
        maximum = np.full(team_rows.max() + 1, -np.inf)
        np.maximum.at(maximum, team_rows, log_weights)
        shifted = log_weights - maximum[team_rows]
        totals = np.bincount(team_rows, np.exp(shifted))
        return shifted - np.log(totals[team_rows])

    def predict(self, rows, team_rows):
        log_weights = self.design.matrix(rows) @ self.coefficients
        return np.exp(self.log_shares(log_weights, team_rows))

    def multipliers(self, rows):
        return self.design.multipliers(rows, self.coefficients).rename(
            columns={"position_role": "position_weight", "player_id": "player_multiplier"}
        )
