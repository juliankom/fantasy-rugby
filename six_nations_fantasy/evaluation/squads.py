"""Select and display legal fantasy squads."""

import pandas as pd
from pulp import COIN_CMD, LpStatus, LpStatusInfeasible, LpStatusOptimal, PULP_CBC_CMD, value

from six_nations_fantasy.ilp import setup_problem


def squad_solver():
    solver = COIN_CMD(msg=False)
    return solver if solver.available() else PULP_CBC_CMD(msg=False)


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
