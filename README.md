# Fantasy Rugby Predictor
A project exploring various statistical models to predicting players' performances in (fantasy) six nations. 

# Data
Currently, we have 69782 performances from 3031 unique players across the major
leagues (URC, Top14, Prem) and competitions (Champions Cup, Six Nations). 
Most statistics relevant for calculating fantasy score, such as meters carried, tackles, 
etc. are tracked.

# Aims
To explore different statistical models predicting players' fantasy scores in a given
game (best would be specifically for Six Nations games). The baseline
here is the naive predictor taking the mean of all of a players previous performances.

Currently implemented are:
- The naive mean
- The mean weighted to favour recent performances (using exponential decay)
- A Bayesian model assuming that fantasy scores are log-normally distributed, allowing us to calculate a shrinkage factor
    for players with few minutes and few appearances

Future: 
- Cleaner testing pipeline 
- Models made specifically to predict individual stats like tackles 
- Random forest models
- Gradient boosting

### Notes
Make sure to ``.venv/bin/activate``.
