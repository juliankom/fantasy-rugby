"""Score Fantasy Rugby Geek shortlists and a pre-match reconstruction for each round.

published_points in frg_picks_2026.csv comes from The Oval Office:
https://ovalofficerugby.com/data/stats_feed_six-nations-2026.json
Article list order is a reconstruction tie-break, not an asserted expert ranking.
"""

import argparse
from pathlib import Path

import pandas as pd
from pulp import LpStatus, LpStatusOptimal, PULP_CBC_CMD, lpSum, value

from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.ilp import setup_problem
from test_selection import ROUND_STARTS, add_name_keys, read_matchday_csv, round_candidates

HERE = Path(__file__).resolve().parent
CAPTAIN = "Recommended Captain Picks"
SUB = "Supersub Strategy"
SECONDARY = {"Bargain buys on the Official game", "Round 5 Differentials"}


def reconstruct(pool, picks):
    # Only published options can occupy their respective roles. Prefer positional
    # recommendations to other tips, then earlier entries in each printed list.
    candidates = pool.dropna(subset=["cost"]).assign(score=0)
    problem = setup_problem(candidates)
    terms = []
    by_player = picks.groupby(["team", "name_key"])
    for player in candidates.itertuples():
        key = (player.team, player.name_key)
        options = picks.iloc[:0] if key not in by_player.groups else by_player.get_group(key)
        field = options.loc[~options.section.isin([CAPTAIN, SUB])]
        main = field.loc[~field.section.isin(SECONDARY)]
        if field.empty or player.is_sub:
            problem += player.selected_var == 0
        else:
            order = (main if len(main) else field).list_order.min()
            terms.append(player.selected_var * (1000 * int(bool(len(main))) - int(order)))
        for section, variable in [(CAPTAIN, player.captain_var), (SUB, player.supersub_var)]:
            ranks = options.loc[options.section == section, "list_order"]
            if ranks.empty or (section == SUB and not player.is_sub):
                problem += variable == 0
            else:
                terms.append(-int(ranks.min()) * variable)
    problem.setObjective(lpSum(terms))
    status = problem.solve(PULP_CBC_CMD(msg=False))
    if status != LpStatusOptimal:
        raise ValueError(f"Cannot reconstruct a legal recommended squad: {LpStatus[status]}")
    squad = candidates.loc[candidates.selected_var.map(value).eq(1)
                           | candidates.supersub_var.map(value).eq(1)].copy()
    squad["multiplier"] = 1 + squad.captain_var.map(value) + 2 * squad.supersub_var.map(value)
    return squad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        help="Use project simple_score from match Parquets instead of published points")
    args = parser.parse_args()
    roster = read_matchday_csv(HERE / "six_nations_2026_fantasy_prices.csv")
    picks = add_name_keys(pd.read_csv(HERE / "frg_picks_2026.csv"))
    if args.data_dir:
        tournament = load_dataset(args.data_dir).filter_competitions(["Six Nations"])
        players = pd.read_parquet(args.data_dir / "players.parquet")
    keys = ["team", "name_key"]
    scores={}
    for number, tips in picks.groupby("round"):
        pool = roster.loc[roster["round"] == number]
        if args.data_dir:
            start, end = ROUND_STARTS[number-1:number+1]
            matches = tournament.matches
            ids = matches.loc[matches.date.between(start, end, inclusive="left"), "match_id"]
            pool = round_candidates(number, pool, tournament.select_matches(ids), players)
            tips = tips.merge(pool[keys + ["actual"]], how="left", on=keys, validate="many_to_one")
        else:
            tips = tips.assign(actual=tips.published_points)
            outcomes = tips[keys + ["actual"]].drop_duplicates()
            pool = pool.merge(outcomes, on=keys, how="left", validate="one_to_one")
        squad = reconstruct(pool, tips)
        if squad.actual.isna().any():
            raise ValueError(f"Round {number}: missing scores for selected players")
        squad["points"] = squad.actual * squad.multiplier
        #print(f"\nRound {number} recommendations (base points)")
        #print(tips[["section", "source_name", "actual"]].to_string(index=False, na_rep="unavailable"))
        print(f"Round {number} squad: {squad.points.sum():g} points")
        scores[number] = squad.points.sum()
        #print(squad[["name", "team", "nominal_pos", "actual", "multiplier", "points"]].to_string(index=False))
    frg_estimated_scores = pd.DataFrame(scores.values(), columns=["score"])
    frg_estimated_scores["round_number"] = list(range(1, 6))
    frg_estimated_scores.to_parquet(HERE / "frg_estimated_scores.parquet")

if __name__ == "__main__":
    main()
