# TODO these should be full-format for compatibility: Second-row, ...

# (FB) Fullbacks, (W)ings, (C)enters, (FH) Fly halves, (SH) Scrum halves
#backs_positions = set(['FB', 'W', 'C', 'FH', 'SH'])
backs_positions = set(['Scrum-half', 'Fly-half', 'Centre', 'Back Three'])

# (P)rops, (H)ookers, (L)ocks, (FL)ankers, (N8) Number 8s
forwards_positions = set(['P', 'H', 'L', 'FL', 'N8'])

# TODO should try to move away from per-row operations and focus on 
# working with the columns directly (maybe)?
#
# TODO for now, I just ignore the difference in scoring (hopefully does not affect
# optimal team too much)
def score_common(player): 
    return 10 * player['tries'] \
        + 4 * player['assists'] \
        + 2 * player['conversions'] \
        + 3 * player['penalties_kicked'] \
        + 5 * player['drop_goals'] \
        + 2 * player['defenders_beaten'] \
        + player['meters_carried'] // 10 \
        + player['tackles'] \
        + 7 * player['lineout_steals'] \
        + 2 * player['offloads'] \
        - 5 * player['yellow_cards'] \
        - 8 * player['red_cards']

def score_back(player):
    return 10 * player['tries'] + score_common(player)

def score_forward(player):
    return 15 * player['tries'] + score_common(player)

def score(player):
    return score_common(player)

    #if player['nominal_pos'] in backs_positions:
    #    return score_back(player)
    #else:
    #    return score_forward(player)
