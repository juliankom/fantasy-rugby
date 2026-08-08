"""Estimate a player's future performance based on the historical mean of this
players performance with some shrinkage towards the global mean 
"""

class BayesShrink:
    def __init__(self, f, performances, current_date, decay_factor=0.9):
        self.performances = performances.copy(deep=False)

        # mean of all scores in the dataset, this is used as a default
        # value for players not previously encountered
        self.global_mean = self.performances.apply(f, axis=1).mean()

        self.performances.loc[:, 'score'] = self.performances.apply(f, axis=1)

        # calculate weights based on date of match
        self.performances.loc[:, 'weight'] = self.performances.apply(
                lambda p: decay_factor**((current_date - p['date']).days / 7), axis=1)

        self.substitute_factors = substitute_factors(self.performances)

        #TODO we might not always be passing in the nominal positions 
        #TODO what if a play plays multiple positions? Then just take 
        # the data of a player at that position!?
        self.position_data = self.performances
            .groupby('position')
            .apply(lambda grp: grp['score'].mean())

        # per-player estimates
        self.estimates = self.performances.groupby('name').apply(estimate_player_score)

    def estimate(self, player_name):
        # estimate for a given player their mean points thus far, 
        # if a player does not appear in the database so far, assign
        # the global mean as a prediction
        return self.estimates.get(player_name, self.global_mean)
