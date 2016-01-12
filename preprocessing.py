import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing as pre
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.calibration import CalibratedClassifierCV

def convert_date_data(date_data):
    date = pd.to_datetime(pd.Series(date_data))
    return np.vstack([date.dt.year, date.dt.month,
                      date.dt.day, date.dt.dayofweek])

from collections import Counter
def extract_object_data(column, cat2vectors = True, counts = None):
  if counts is None:
    counts = Counter(column)
  else:
    new_counts = column.unique()
    new_values = list(set(new_counts).difference(counts.keys()))
    if len(new_values) > 0:
      print 'new values', new_values
      column = column.replace(new_values, np.nan)
      print 'new values', set(column.unique()).difference(counts)


  column = column.fillna(max(filter(lambda x: x != np.nan, counts.keys()),
                             key = lambda x: counts[x]))
  if cat2vectors:
    c2i = dict((c, i) for i,c in enumerate(counts.keys()))
    ic = np.eye(len(c2i))
    ti = map(lambda x: c2i[x], column)
    data = ic[ti]
    # data = pd.get_dummies(column).values
    if data.shape[1] == 1:
      print 'constant data'
      return []
  else:
    c2i = dict((c, i) for i,c in enumerate(counts.keys(), start=1))
    if len(c2i) == 1:
      print 'constant data'
      return []
    data = np.asarray([c2i[k] for k in column])
  return data.T, counts

def extract_numeric_data(column, normalize_numeric = True, mean_std = None):
  column = column.fillna(column.median())
  if len(column.unique()) == 1:
    return []

  if normalize_numeric:
    if mean_std is None:
      mean_std = [column.mean(),column.std()]
    column = column - mean_std[0]
    column = column/mean_std[1]
  return column.values, mean_std

def extract_ordinal_data(column, normalize_numeric = False, mean_std = None):
    return extract_numeric_data(column, normalize_numeric, mean_std = mean_std)


def convert_data(dataframe, cat2vectors = True, normalize_numeric = True, test = None):
    y = dataframe.QuoteConversion_Flag.values
    X, Xtest = [], []
    X.append(convert_date_data(dataframe['Original_Quote_Date']))
    if test is not None:
      Xtest.append(convert_date_data(test['Original_Quote_Date']))
    columns_to_skip = ['Original_Quote_Date', 'QuoteConversion_Flag',
                      'QuoteNumber']
    for c in dataframe.columns:
        if c in columns_to_skip:
            print c, 'skipped'
            continue
        test_column = None if test is None else test[c]
        if dataframe[c].dtype == object:
            rv = extract_object_data(dataframe[c], cat2vectors)
            if len(rv) == 0: continue
            X.append(rv[0])
            if test_column is not None:
              Xtest.append(extract_object_data(test_column, cat2vectors, counts = rv[1])[0])
        elif c.endswith('A') or c.endswith('B'):
            rv= extract_ordinal_data(dataframe[c])
            if len(rv) == 0: continue
            X.append(rv[0])
            if test_column is not None:
              Xtest.append(extract_ordinal_data(test_column)[0])
        else:
            rv = extract_numeric_data(dataframe[c], normalize_numeric)
            if len(rv) == 0: continue
            X.append(rv[0])
            if test_column is not None:
              Xtest.append(extract_numeric_data(test_column, normalize_numeric, mean_std = rv[1])[0])
    X = np.vstack(filter(lambda x: len(x) > 0, X)).T
    if test is not None:
      Xtest = np.vstack(filter(lambda x: len(x) > 0, Xtest)).T
    else:
      Xtest = np.asarray(Xtest)
    return X, y, Xtest

def load_data(get_nan_feature = True, load_test = False, cat2vectors = False):
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv') if load_test else None
    train.replace([' ', '', -1], np.nan, inplace = True)
    if test is not None:
      test.replace([' ', '', -1], np.nan, inplace = True)
    if get_nan_feature:
        train['NaNCount'] = train.isnull().sum(axis=1)
        if test is not None:
          test['NaNCount'] = test.isnull().sum(axis = 1)
    X, y, Xtest = convert_data(train, cat2vectors = cat2vectors, normalize_numeric = False, test = test)
    print X.shape, y.shape, Xtest.shape
    if load_test:
      return X,y,Xtest, test
    else:
      return X, y
