# kaggle
When to use a boosted trees model?
Different kinds of models have different advantages. The boosted trees model is
very good at handling tabular data with numerical features, or categorical
features with fewer than hundreds of categories.

One important note is that tree based models are not designed to work with very
sparse features. When dealing with sparse input data (e.g. categorical features
with large dimension), we can either pre-process the sparse features to generate
numerical statistics, or switch to a linear model, which is better suited for
such scenarios.

TODO:
use gridsearch and votingclassifier
http://scikit-learn.org/stable/modules/ensemble.html
https://medium.com/@chris_bour/6-tricks-i-learned-from-the-otto-kaggle-challenge-a9299378cd61#.sjtyovd6g
