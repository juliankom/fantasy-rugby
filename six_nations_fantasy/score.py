

# TODO these should be full-format for compatibility: Second-row, ...

# (FB) Fullbacks, (W)ings, (C)enters, (FH) Fly halves, (SH) Scrum halves
backs_positions = set(['FB', 'W', 'C', 'FH', 'SH'])

# (P)rops, (H)ookers, (L)ocks, (FL)ankers, (N8) Number 8s
forwards_positions = set(['P', 'H', 'L', 'FL', 'N8'])

# TODO should try to move away from per-row operations and focus on 
# working with the columns directly (maybe)?
def score_common(player): 
    return 4 * player['assists'] \
        + 2 * player['conversions'] \
        + 3 * player['penalties_kicked'] \
        + 5 * player['drop_goals'] \
        + 2 * player['defenders_beaten'] \
        + player['meters_carried'] // 10 \
        + player['tackles'] \
        + 7 * player['lineout_steals'] \
        - 5 * player['yellow_cards'] \
        - 8 * player['red_cards']

def score_back(player):
    return 10 * player['tries'] + score_common(player)

def score_forward(player):
    return 15 * player['tries'] + score_common(player)

def score(player):
    if player['position'] in backs_positions:
        return score_back(player)
    elif player['position'] in forwards_positions:
        return score_forward(player)
    else:
        # TODO what to do about substitutes?
        # Well, this again depends on the database of _nominal_ positions,
        # which we don't have at this point...
        return 0

