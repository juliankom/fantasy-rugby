"""Penalized likelihoods for team counts and within-lineup allocations."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack


class Categories:
    """One indicator per known level. An unseen level contributes zero on the log scale."""

    def __init__(self, frame, columns):
        self.levels = {col: pd.Index(frame[col].unique()) for col in columns}

    def matrix(self, frame):
        blocks = []
        for col, levels in self.levels.items():
            codes = levels.get_indexer(frame[col])
            rows = np.flatnonzero(codes >= 0)
            blocks.append(csr_matrix((np.ones(len(rows)), (rows, codes[rows])),
                                     shape=(len(frame), len(levels))))
        return hstack(blocks, format="csr")

    def precisions(self, scales):
        # Independent zero-centred normal priors give ridge precision 1 / sigma^2.
        return np.concatenate([np.full(len(levels), 1 / scales[col] ** 2)
                               for col, levels in self.levels.items()])


def minimize_loss(objective, size):
    result = minimize(objective, np.zeros(size), jac=True, method="L-BFGS-B",
                      options={"maxiter": 600, "ftol": 1e-11})
    if not result.success:
        raise RuntimeError(f"Share model fit failed: {result.message}")
    return result.x


class TeamRate:
    def __init__(self, frame, counts, tau):
        # Use the pooled training mean as offset. An unseen competition starts here.
        self.base_rate = counts.mean()
        scales = {"competition": 2.0, "home": 0.5, "team": 0.5}
        if tau > 0:
            scales["opponent"] = tau
        self.categories = Categories(frame, scales)
        X = self.categories.matrix(frame)
        precision = self.categories.precisions(scales)

        def objective(beta):
            # log(mu) = log(base_rate) + competition + venue + team + opponent.
            # Poisson negative log likelihood, dropping terms independent of beta.
            eta = X @ beta
            mu = self.base_rate * np.exp(eta)
            loss = (mu - counts * eta).sum() + 0.5 * np.sum(precision * beta**2)
            gradient = X.T @ (mu - counts) + precision * beta
            return loss, gradient

        self.beta = minimize_loss(objective, X.shape[1]) if self.base_rate else np.zeros(X.shape[1])

    def predict(self, frame):
        return self.base_rate * np.exp(self.categories.matrix(frame) @ self.beta)


def choose_tau(teams, counts, earlier, shared_tau):
    """Regularize the component scale using a chronological predictive likelihood."""
    if not earlier.any() or earlier.all() or counts[earlier].sum() == 0:
        return shared_tau
    observed = counts[~earlier]
    candidates = [shared_tau, shared_tau / 2, shared_tau * 2]
    losses = []
    for tau in candidates:
        model = TeamRate(teams.loc[earlier], counts[earlier], tau)
        mu = model.predict(teams.loc[~earlier])
        # Predictive Poisson loss plus a N(0, log(2)^2) penalty on log(tau/shared_tau).
        penalty = 0.5 * (np.log(tau / shared_tau) / np.log(2)) ** 2
        losses.append((mu - observed * np.log(mu)).sum() + penalty)
    return candidates[np.argmin(losses)]


class Allocation:
    def __init__(self, rows, groups, totals, counts):
        scales = {"position_role": 2.0, "player_id": 0.75}
        self.categories = Categories(rows, scales)
        X = self.categories.matrix(rows)
        precision = self.categories.precisions(scales)

        def objective(beta):
            # Conditional multinomial log likelihood: for each team sheet g,
            # sum_i y_i log(p_i), where p_i = exp(x_i beta) / sum_g exp(x_j beta).
            shares = self.probabilities(X @ beta, groups)
            loss = -np.sum(counts * np.log(shares)) + 0.5 * np.sum(precision * beta**2)
            gradient = X.T @ (totals[groups] * shares - counts) + precision * beta
            return loss, gradient

        self.beta = minimize_loss(objective, X.shape[1]) if counts.any() else np.zeros(X.shape[1])

    @staticmethod
    def probabilities(log_weights, groups):
        maximum = np.full(groups.max() + 1, -np.inf)
        np.maximum.at(maximum, groups, log_weights)
        weights = np.exp(log_weights - maximum[groups])
        return weights / np.bincount(groups, weights)[groups]

    def predict(self, rows, groups):
        return self.probabilities(self.categories.matrix(rows) @ self.beta, groups)
