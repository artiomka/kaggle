
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
import classifier as cl_kaggle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing as pre
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

def find_xgb_best_parameters(test_size = 0.2, n_iter_search = 20):
  X, y = pr_kaggle.load_data(cat2vectors=True)
  Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
    test_size=test_size, random_state=36)
  param_dist = {
  "n_estimators" : [50, 100, 250, 500]
  "max_depth": [3, 10, 15, 20],
  "learning_rate": [0.01, 0.1, 0.05],
  "subsample": [0.5, 1.0, 0.85, 0.7],
  "colsample_bytree": [1.0, 0.5, 0.66, 0.75, 0.9]}
  clf = xgb.XGBClassifier()
  random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
   n_iter=n_iter_search, n_jobs = 4)

  random_search.fit(Xtrain, ytrain)
  print("RandomizedSearchCV took %.2f seconds for %d candidates"
    " parameter settings." % ((time() - start), n_iter_search))
  report(random_search.grid_scores_)

  print 'training', clf.score(Xtrain, ytrain)
  print 'training', clf.score(Xtest, ytest)
  return random_search

