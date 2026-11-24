"""Fit a Poisson regression to each team statistic."""

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


def minimize_loss(objective, curvature, regression):
    """Rescale the coefficients so frequently and rarely seen categories fit together."""
    # A step in a coefficient seen in thousands of lineups changes the loss
    # much more than the same step for a player seen once. Divide the step
    # by the square root of estimated curvature to bring those changes closer.
    # This is a change of coordinates for the optimizer. The loss and ridge
    # penalty still use coefficients in their original units.
    step_scale = 1 / np.sqrt(curvature)

    def scaled_objective(parameters):
        loss, gradient = objective(parameters * step_scale)
        return loss, gradient * step_scale

    # L-BFGS-B uses the supplied gradient. ftol tests relative improvement in loss.
    result = minimize(scaled_objective, np.zeros(len(curvature)), jac=True, method="L-BFGS-B",
                      options={"maxiter": 600, "ftol": 1e-11})
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        gradient = np.linalg.norm(result.jac, ord=np.inf)
        raise RuntimeError(
            f"{regression}: {result.message} (iterations={result.nit}, max scaled gradient={gradient:.3g})"
        )
    return result.x * step_scale


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

        # At zero coefficients every team starts at base_rate. With binary
        # features the Poisson curvature per coefficient is its total expected count.
        curvature = self.base_rate * np.asarray(design.sum(axis=0)).ravel() + precision
        self.coefficients = (
            minimize_loss(objective, curvature, "team total") if self.base_rate else np.zeros_like(precision)
        )

    def predict(self, frame):
        return self.base_rate * np.exp(self.design.matrix(frame) @ self.coefficients)

    def multipliers(self, frame):
        factors = self.design.multipliers(frame, self.coefficients)
        return factors.rename(columns=lambda col: f"{col}_multiplier")
