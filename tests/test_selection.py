"""Compare fantasy squads in a rolling backtest of the 2026 Six Nations."""

import argparse
import re
import unicodedata
from math import log
from pathlib import Path

import pandas as pd
from pulp import COIN_CMD, LpStatus, LpStatusInfeasible, LpStatusOptimal, PULP_CBC_CMD, value

from six_nations_fantasy.data import load_dataset
from six_nations_fantasy.ilp import setup_problem
from six_nations_fantasy.models import HistoricalMean, PositionShrinkageMean, RecencyWeightedMean
from six_nations_fantasy.score import score_performances, simple_score


ROUND_STARTS = pd.to_datetime(
    ["2026-02-05", "2026-02-14", "2026-02-21", "2026-03-06", "2026-03-14", "2026-04-01"]
)
POSITIONS = {
    "hooker": "Hooker", "prop": "Prop", "second_row": "Second-row",
    "back_row": "Back-row", "scrum_half": "Scrum-half", "fly_half": "Fly-half",
    "centre": "Centre", "back_three": "Back Three",
}
NAME_ALIASES = {
    ("Ireland", "tomclarkson"): "thomasclarkson",
    ("Ireland", "tomfarrell"): "thomasfarrell",
    ("Italy", "nachobrex"): "juanignaciobrex",
    ("France", "noahnene"): "noahtisienene",
    ("Scotland", "freddydouglas"): "freddiedouglas",
    ("Wales", "gabehamerwebb"): "gabrielhamerwebb",
    ("Wales", "danedwards"): "danieledwards",
    ("Italy", "davidodiase"): "davideodiase",
    ("England", "finsmith"): "finlaysmith",
    ("England", "sebatkinson"): "sebastienatkinson",
}


# Workbook roster entries with no appearance in the processed match records.
# (These can be players who were initially selected but have been injured in the lead-up.)
NON_APPEARANCES = {
    (1, "England", "immanuelfeyiwaboso"),
    (3, "France", "matthieujalibert"),
    (3, "Ireland", "jackconan"),
    (4, "England", "tomcurry"),
}


# internal helper key used to later find the correct player_ids for the players in the csv
def name_key(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def add_name_keys(frame):
    return frame.assign(name_key=[
        NAME_ALIASES.get((team, key), key)
        for team, key in zip(frame["team"], frame["name"].map(name_key))
    ])


def read_matchday_csv(filename):
    roster = pd.read_csv(filename).rename(columns={"player": "name", "price": "cost"})
    roster = roster.loc[roster["year"] == 2026].copy()
    if set(roster["round"]) != set(range(1, len(ROUND_STARTS))):
        raise ValueError("Expected a roster for each 2026 round")
    roster["nominal_pos"] = roster["nominal_position"].map(POSITIONS)
    if roster["nominal_pos"].isna().any() or not roster["selection"].isin(["starter", "bench"]).all():
        raise ValueError("Every roster entry needs a valid fantasy position and selection")
    roster["is_sub"] = roster["selection"] == "bench"
    return add_name_keys(roster)


def round_candidates(round_number, roster, fixtures, players):

    # retrieve the performance data from our stats
    performances = fixtures.player_matches()
    performances["actual"] = score_performances(performances, simple_score)
    performances["team"] = performances["home_team"].where(
        performances["home"], performances["away_team"]
    )

    # try to merge the name_keys derived from the roster csv names with the names 
    # stored in the performance database
    performances = add_name_keys(performances.join(players["name"], on="player_id"))
    candidates = roster.merge(
        performances[["team", "name_key", "match_id", "player_id", "date", "actual"]],
        on=["team", "name_key"], how="left", validate="one_to_one",
    )

    missing = candidates["actual"].isna()
    unmatched = set(zip(candidates.loc[missing, "team"], candidates.loc[missing, "name_key"]))
    expected = {(team, key) for number, team, key in NON_APPEARANCES if number == round_number}
    if unmatched - expected:
        raise ValueError(f"Round {round_number}: unmatched roster entries {sorted(unmatched - expected)}")

    # Retain priced non-appearances in the prediction pool, with their historical
    # player identity and scheduled fixture. Their realized score is zero.
    for index, player in candidates.loc[missing].iterrows():
        player_ids = players.index[players["name"].map(name_key) == player["name_key"]]
        match = fixtures.matches.loc[
            (fixtures.matches["home_team"] == player["team"])
            | (fixtures.matches["away_team"] == player["team"])
        ]
        if len(player_ids) != 1 or len(match) != 1:
            raise ValueError(f"Could not identify player and fixture for {player['name']}")
        candidates.at[index, "player_id"] = player_ids[0]
        candidates.at[index, "match_id"] = match["match_id"].iloc[0]
        candidates.at[index, "date"] = match["date"].iloc[0]
        candidates.at[index, "actual"] = 0
    candidates["player_id"] = candidates["player_id"].astype(players.index.dtype)
    candidates["match_id"] = candidates["match_id"].astype(fixtures.matches["match_id"].dtype)
    return candidates


class InfeasibleSquadError(RuntimeError):
    """The candidate pool cannot fill a legal squad."""


def select_squad(candidates, objective, solver):
    pool = candidates.assign(score=candidates[objective])
    problem = setup_problem(pool)
    status = problem.solve(solver)
    if status == LpStatusInfeasible:
        raise InfeasibleSquadError("No feasible squad")
    if status != LpStatusOptimal:
        raise RuntimeError(f"Squad optimization failed: {LpStatus[status]}")

    selected = pool["selected_var"].map(value).eq(1)
    supersub = pool["supersub_var"].map(value).eq(1)
    squad = pool.loc[selected | supersub].copy()
    squad["captain"] = squad["captain_var"].map(value).eq(1)
    squad["supersub"] = supersub.loc[squad.index]
    multiplier = (1.0 + squad["captain"].astype(float)).where(
        ~squad["supersub"], squad["is_sub"].map({True: 3.0, False: 0.5})
    )
    squad["actual_total"] = squad["actual"] * multiplier
    squad["objective_total"] = squad[objective] * multiplier
    return squad


def single_player_exclusions(candidates, optimal_squad, solver):
    optimal_score = optimal_squad["actual_total"].sum()
    results = []
    for player in optimal_squad.itertuples():
        remaining = candidates.loc[candidates["player_id"] != player.player_id]
        try:
            alternative = select_squad(remaining, "actual", solver)
        except InfeasibleSquadError:
            score = float("nan")
        else:
            score = alternative["actual_total"].sum()

        results.append({
            "excluded": player.name,
            "team": player.team,
            "best_score": score,
            "points_lost": optimal_score - score,
        })
    return pd.DataFrame(results).sort_values("points_lost", ascending=False, kind="stable")


def print_squad(squad):
    pos_order = (['Prop', 'Hooker', 'Prop'] 
                 + 2 * ['Second-row']
                 + 3 * ['Back-row'] 
                 + ['Scrum-half', 'Fly-half']
                 + ['Back Three']
                 + 2 * ['Centre']
                 + 2 * ['Back Three']
    )

    # a table enumerating the positions in the correct order, 
    # also with per-position counts (occurrence)
    slots = pd.DataFrame({"nominal_pos": pos_order})
    slots["occurrence"] = slots.groupby("nominal_pos").cumcount()
    slots["display_order"] = range(len(slots))

    lineup = squad.loc[~squad["supersub"]].copy()
    lineup["occurrence"] = lineup.groupby("nominal_pos").cumcount()
    lineup["captain"] = lineup["captain"].apply(lambda x: "Captain" if x else "")
    lineup = (
        lineup.merge(slots, on=["nominal_pos", "occurrence"], how="left")
        .sort_values("display_order", kind="stable")
        .drop(columns=["occurrence", "display_order"])
    )

    print(lineup[["nominal_pos", "name", "team", "cost", "actual_total", "captain"]]
            .to_string(index=False))

    supersub = squad[squad["supersub"]][["name", "team", "cost", "actual_total"]]
    print(f"Supersub:\n{supersub}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roster", nargs="?", type=Path,
        default=Path(__file__).with_name("six_nations_2026_fantasy_prices.csv"),
    )
    parser.add_argument("--k", type=float, nargs="+", default=[5.0],
                        help="Shrinkage pseudo-appearances; compare several values in one run")
    args = parser.parse_args()
    roster = read_matchday_csv(args.roster)
    data_dir = Path(__file__).resolve().parents[1] / "data" / "rebuild"
    dataset = load_dataset(data_dir)
    tournament = dataset.filter_competitions(["Six Nations"])
    players = pd.read_parquet(data_dir / "players.parquet")
    matches = dataset.matches
    dates = matches["date"]

    models = {
        "Historical mean": HistoricalMean(scorer=simple_score),
        "Recency-weighted mean": RecencyWeightedMean(
            scorer=simple_score, half_life_days=7 * log(0.5) / log(0.85)
        ),
    }
    for k in args.k:
        models[f"Position shrinkage (k={k:g})"] = PositionShrinkageMean(scorer=simple_score, k=k)
    solver = COIN_CMD(msg=False)
    if not solver.available():
        solver = PULP_CBC_CMD(msg=False)
    totals = dict.fromkeys(["Best calculated score", *models], 0.0)

    # the result of all models (and the optimal score) for all rounds
    tournament_result = []

    for round_number, (start, end) in enumerate(zip(ROUND_STARTS, ROUND_STARTS[1:]), 1):

        # identify the matches played in this round
        round_matches = tournament.matches
        round_ids = round_matches.loc[round_matches["date"].between(start, end, inclusive="left"), "match_id"]
        if len(round_ids) != 3:
            raise ValueError(f"Round {round_number}: expected three fixtures, found {len(round_ids)}")
        candidates = round_candidates(
            round_number, roster.loc[roster["round"] == round_number],
            tournament.select_matches(round_ids), players,
        )

        print(f"\nRound {round_number}")
        unpriced = candidates["cost"].isna()
        if unpriced.any():
            print(f"Excluded for missing price: {', '.join(candidates.loc[unpriced, 'name'])}")
        training = dataset.select_matches(matches.loc[dates < start, "match_id"])
        objectives = {"Best calculated score": "actual"}
        prediction_fixtures = candidates[
            ["match_id", "player_id", "date", "team", "shirt_number"]
        ].rename(columns={"shirt_number": "number"}).assign(is_substitute=candidates["is_sub"])
        for label, model in models.items():
            model.fit(training)
            candidates[label] = model.predict(prediction_fixtures.copy())
            objectives[label] = label
        candidates = candidates.loc[~unpriced].copy()

        # for each objective (actual score, or score predicted by a model), compute the optimal squad
        round_results = {}
        for label, objective in objectives.items():
            squad = select_squad(candidates, objective, solver)
            actual = squad["actual_total"].sum()
            totals[label] += actual
            round_results[label] = actual
            print(f"{label}: cost {squad['cost'].sum():.1f}, "
                  f"objective {squad['objective_total'].sum():.1f}, calculated actual {actual:.1f}")
            print_squad(squad)
            if objective == "actual":
                print("\nHindsight score after each single-player exclusion")
                exclusions = single_player_exclusions(candidates, squad, solver)
                print(exclusions.to_string(index=False, na_rep="infeasible"))

        tournament_result.append(round_results)

    print("\nTournament totals (calculated actual)")
    tournament_optimal = totals["Best calculated score"]
    for label, total in totals.items():
        print(f"{label}: {total:.1f} ({(total / tournament_optimal)*100:.1f}%)")

    result = pd.DataFrame(tournament_result)
    try:
        frg = pd.read_parquet(Path(__file__).resolve().parent / "frg_estimated_scores.parquet")
        result["FRG score"] = frg["score"]
    except:
        pass

    result.loc["total"] = result.sum(axis=0)
    print(result)
if __name__ == "__main__":
    main()
