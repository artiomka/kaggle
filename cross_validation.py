

# coding: utf-8

import xgboost
import numpy as np

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

import xgboost as xgb
import preprocessing as pr_kaggle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing as pre
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier


import xgboost as xgb
import preprocessing as pr_kaggle
import classification as cl_kaggle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing as pre
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def find_xgb_best_parameters(test_size = 0.2, n_iter_search = 20, X = None, y=None):
  if X is None or y is None:
    X, y = pr_kaggle.load_data(cat2vectors=True)
  Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
    test_size=test_size, random_state=36)
  param_dist = {
  "n_estimators" : [50, 100, 250, 500],
  "max_depth": [10,5,15],
  "learning_rate": [0.01, 0.1, 0.0333],
  "subsample": [0.5, 1.0, 0.80],
  #"gamma": [0,0.01],
  #"min_child_weight": [0.5, 1],
  "colsample_bytree": [1.0, 0.5, 0.8, 0.9]}
  start = time()
  clf = xgb.XGBClassifier()
  random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
   n_iter=n_iter_search, n_jobs = 1)


  print Xtrain.shape
  random_search.fit(Xtrain, ytrain)
  print("RandomizedSearchCV took %.2f seconds for %d candidates"
    " parameter settings." % ((time() - start), n_iter_search))
  report(random_search.grid_scores_)

  print 'training', random_search.score(Xtrain, ytrain)
  print 'testing', random_search.score(Xtest, ytest)
  return random_search

