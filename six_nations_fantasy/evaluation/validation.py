"""Price-free validation diagnostics for starting XV selections."""

from math import ceil

import numpy as np
import pandas as pd

from six_nations_fantasy.ilp import POSITION_COUNTS
from six_nations_fantasy.models.position_shrinkage import STARTING_POSITIONS


def subset_scores(fixtures, predictions, rng, draws=1000):
    """Reselect each model's XV on identical random subsets of each position."""
    starters = fixtures.loc[~fixtures["is_substitute"].fillna(False)].sort_values(["player_id", "match_id"])
    positions = pd.to_numeric(starters["number"]).map(STARTING_POSITIONS)
    totals = pd.DataFrame(0., index=range(draws), columns=predictions.keys())
    for position, group in starters.groupby(positions):
        slots = POSITION_COUNTS[position]
        keep = max(slots, ceil(0.75 * len(group)))
        # Each row is one draw without replacement. All models see the same mask.
        random_order = rng.random((draws, len(group))).argsort(axis=1)
        available = random_order < keep
        actual = group["actual"].to_numpy()
        for label, predicted in predictions.items():
            # Stable descending forecast order preserves the player-ID tie breaker.
            order = np.argsort(-predicted.loc[group.index].to_numpy(), kind="stable")
            eligible = available[:, order]
            chosen = eligible & (eligible.cumsum(axis=1) <= slots)
            totals[label] += (chosen * actual[order]).sum(axis=1)
    return totals


def plot_calibration(selected, path):
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    groups = selected.groupby("model", sort=False)
    fig = Figure(figsize=(7, 4 * len(groups) + 1), layout="constrained")
    axes = fig.subplots(len(groups), 1, sharex=True, sharey=True, squeeze=False)
    limits = [selected[["actual", "predicted"]].min().min(), selected[["actual", "predicted"]].max().max()]
    for ax, (model, players) in zip(axes.flat, groups):
        for i, position in enumerate(POSITION_COUNTS):
            group = players.loc[players.position.eq(position)]
            ax.scatter(group.predicted, group.actual, s=15, alpha=.5, color=f"C{i}", label=position)
        ax.plot(limits, limits, "--", color="grey")
        # Equal-frequency bands; keep identical forecasts in the same band.
        bands = pd.qcut(players.predicted, 4, duplicates="drop")
        for _, band in players.groupby(bands, observed=True):
            x, y = band.predicted.mean(), band.actual.mean()
            ax.plot(x, y, "kD")
            ax.annotate(f"n={len(band)}", (x, y), xytext=(3, 6), textcoords="offset points")
        if players.predicted.nunique() > 1:
            slope, intercept = np.polyfit(players.predicted, players.actual, 1)
            x = np.array([players.predicted.min(), players.predicted.max()])
            ax.plot(x, intercept + slope*x, color="black", label=f"Fit (slope {slope:.2f})")
        ax.set(title=model, ylabel="Actual points")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [Line2D([], [], color="grey", linestyle="--"), Line2D([], [], color="k", marker="D", linestyle="")],
                  labels + ["Perfect calibration", "Band mean"], fontsize=7, ncol=3)
    axes[-1, 0].set_xlabel("Predicted points")
    fig.savefig(path, dpi=140)
