# TODO should try to move away from per-row operations and focus on 
# working with the columns directly (maybe)?
def score(player, try_points=15): 
    return try_points * player['tries'] \
        + 4 * player['try_assists'] \
        + 2 * player['conversions'] \
        + 3 * player['penalty_goals'] \
        + 5 * player['drop_goals'] \
        + 2 * player['defenders_beaten'] \
        + player['carried_meters'] // 10 \
        + player['completed_tackles'] \
        + 5 * player['turnovers_won'] \
        + 2 * player['offloads'] \
        - 5 * player['yellow_cards'] \
        - 8 * player['red_cards']

def score_back(player):
    return score(score, try_points=10)

def score_forward(player):
    return score(player)