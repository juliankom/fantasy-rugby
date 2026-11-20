"""Sample uniform legal squads for each round and save their scores for later comparisons."""

import argparse
from collections import Counter
from pathlib import Path
from random import Random

import pandas as pd
from pulp import COIN_CMD, PULP_CBC_CMD

from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.ilp import BUDGET, MAX_PER_TEAM, POSITION_COUNTS
from test_selection import ROUND_STARTS, read_matchday_csv, round_candidates, select_squad


def sample_scores(pool, count, rng):
    """Independent draws with replacement over legal XV, captain and supersub choices."""
    costs = pool["cost"].tolist()
    teams = pool["team"].tolist()
    scores = pool["actual"].tolist()
    sub_scores = pool["actual"] * pool["is_sub"].map({True: 3.0, False: 0.5})
    sub_scores = sub_scores.tolist()
    groups = pool.groupby("nominal_pos").indices
    positions = [(list(groups[pos]), size) for pos, size in POSITION_COUNTS.items()]
    accepted = []
    while len(accepted) < count:
        xv = []
        for players, size in positions:
            xv.extend(rng.sample(players, size))
        # Every XV has the same number of remaining supersub candidates.
        sub = rng.randrange(len(pool))
        if sub in xv:
            continue
        squad = [*xv, sub]
        if sum(costs[i] for i in squad) > BUDGET:
            continue
        if max(Counter(teams[i] for i in squad).values()) > MAX_PER_TEAM:
            continue
        captain = rng.choice(xv)
        accepted.append(sum(scores[i] for i in xv) + scores[captain] + sub_scores[sub])
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000, help="Accepted squads per round")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("sampled_squad_scores.csv"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    roster = read_matchday_csv(Path(__file__).with_name("six_nations_2026_fantasy_prices.csv"))
    data_dir = Path(__file__).resolve().parents[1] / "data" / "rebuild"
    tournament = load_dataset(data_dir).filter_competitions(["Six Nations"])
    players = pd.read_parquet(data_dir / "players.parquet")
    solver = COIN_CMD(msg=False)
    if not solver.available():
        solver = PULP_CBC_CMD(msg=False)
    rng = Random(args.seed)
    rounds = []
    for number, (start, end) in enumerate(zip(ROUND_STARTS, ROUND_STARTS[1:]), 1):
        matches = tournament.matches
        ids = matches.loc[matches["date"].between(start, end, inclusive="left"), "match_id"]
        if len(ids) != 3:
            raise ValueError(f"Round {number}: expected three fixtures, found {len(ids)}")
        pool = round_candidates(number, roster.loc[roster["round"] == number],
                                tournament.select_matches(ids), players).dropna(subset=["cost"])
        # The optimal solve also verifies feasibility before rejection sampling.
        optimum = select_squad(pool, "actual", solver)["actual_total"].sum()
        scores = pd.Series(sample_scores(pool, args.samples, rng))
        rounds.append(pd.DataFrame({"round": number, "score": scores, "optimum": optimum}))
        print(f"Round {number} (optimum {optimum:g}, {len(scores)} sampled squads)")
        for threshold in (0.6, 0.7, 0.8):
            hits = scores.ge(threshold * optimum).sum()
            print(f"  Reached {threshold:.0%}: {hits}/{len(scores)} ({hits / len(scores):.2%})")
    pd.concat(rounds, ignore_index=True).to_csv(args.output, index=False)
    print(f"Saved sampled scores to {args.output}")


if __name__ == "__main__":
    main()
