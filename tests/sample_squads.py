"""Sample legal squads with a starters-only XV and a bench-only supersub."""

import argparse
from collections import Counter
from pathlib import Path
from random import Random

import pandas as pd
from pulp import LpStatus, LpStatusOptimal

from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.ilp import BUDGET, MAX_PER_TEAM, POSITION_COUNTS, setup_problem
from six_nations_fantasy.data.load import DATA_DIR
from six_nations_fantasy.data.fantasy_fixtures import fixture_path, with_actual_scores
from six_nations_fantasy.evaluation.squads import select_squad, squad_solver


def sample_scores(scored_fixtures, count, rng):
    """Independent draws with replacement over legal XV, captain and supersub choices."""
    costs = scored_fixtures["cost"].tolist()
    teams = scored_fixtures["team"].tolist()
    scores = scored_fixtures["actual"].tolist()
    bench = scored_fixtures["is_sub"].to_numpy()
    substitutes = [index for index, is_sub in enumerate(bench) if is_sub]
    position_values = scored_fixtures["nominal_pos"].to_numpy()
    groups = {}
    for index, position in enumerate(position_values):
        if not bench[index]:
            groups.setdefault(position, []).append(index)
    positions = [(groups[pos], size) for pos, size in POSITION_COUNTS.items()]
    accepted = []
    while len(accepted) < count:
        xv = []
        for players, size in positions:
            xv.extend(rng.sample(players, size))
        sub = rng.choice(substitutes)
        squad = [*xv, sub]
        if sum(costs[i] for i in squad) > BUDGET:
            continue
        if max(Counter(teams[i] for i in squad).values()) > MAX_PER_TEAM:
            continue
        captain = rng.choice(xv)
        accepted.append(sum(scores[i] for i in xv) + scores[captain] + 3 * scores[sub])
    return accepted


def check_sampling_feasibility(scored_fixtures, solver, round_number):
    # A feasible unrestricted optimum does not guarantee a feasible restricted draw.
    candidates = scored_fixtures.assign(score=0)
    problem = setup_problem(candidates)
    for player in candidates.itertuples():
        if player.is_sub:
            problem += player.selected_var == 0
        else:
            problem += player.supersub_var == 0
    status = problem.solve(solver)
    if status != LpStatusOptimal:
        raise RuntimeError(
            f"Round {round_number}: starters-only XV and bench-only supersub "
            f"feasibility check returned {LpStatus[status]}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000, help="Accepted squads per round")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("sampled_squad_scores.csv"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    roster = pd.read_parquet(fixture_path(DATA_DIR, 2026))
    dataset = load_dataset(DATA_DIR)
    solver = squad_solver()
    rng = Random(args.seed)
    rounds = []
    for round_number, entries in roster.groupby("round"):
        scored_fixtures = with_actual_scores(entries, dataset.select_matches(entries["match_id"])).dropna(subset=["cost"])
        check_sampling_feasibility(scored_fixtures, solver, round_number)
        # Keep thresholds relative to the unrestricted backtest optimum.
        optimum = select_squad(scored_fixtures, "actual", solver)["actual_total"].sum()
        scores = pd.Series(sample_scores(scored_fixtures, args.samples, rng))
        rounds.append(pd.DataFrame({"round": round_number, "score": scores, "optimum": optimum}))
        print(f"Round {round_number} (optimum {optimum:g}, {len(scores)} sampled squads)")
        for threshold in (0.6, 0.7, 0.8):
            hits = scores.ge(threshold * optimum).sum()
            print(f"  Reached {threshold:.0%}: {hits}/{len(scores)} ({hits / len(scores):.2%})")
    pd.concat(rounds, ignore_index=True).to_csv(args.output, index=False)
    print(f"Saved sampled scores to {args.output}")


if __name__ == "__main__":
    main()
