"""Estimate a player's future performance based on the historical mean of this
players performance.
"""

class NaiveMean:
    def __init__(self, f, performances):
        # mean of all scores in the dataset
        self.global_mean = performances.apply(f, axis=1).mean()

        # per-player means
        self.estimates = performances.groupby('name').apply(
                lambda grp: grp.apply(f, axis=1).mean())

    def estimate(self, player_name):
        # estimate for a given player their mean points thus far, 
        # if a player does not appear in the database so far, assign
        # the global mean as a prediction
        return self.estimates.get(player_name, self.global_mean)
