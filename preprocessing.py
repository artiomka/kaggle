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
def extract_object_data(row, cat2vectors = True):
    counts = Counter(row)
    row = row.fillna(max(filter(lambda x: x != np.nan, counts.keys()), 
               key = lambda x: counts[x]))
    if cat2vectors:
        data = pd.get_dummies(row).values
    else:
        c2i = dict((c, i) for i,c in enumerate(counts.keys(), start=1))
        data = np.asarray([c2i[k] for k in row])
    if data.shape[1] == 1:
        print 'constant data'
        return []
    return data.T

def extract_numeric_data(row, normalize_numeric = True):
    row = row.fillna(row.median())
    if len(row.unique()) == 1:
        return []

    if normalize_numeric:
        row = row - row.mean()
        row = row/row.std()
    return row.values

def extract_ordinal_data(row):
    return extract_numeric_data(row)


def convert_data(dataframe, cat2vectors = True, normalize_numeric = True):
    y = dataframe.QuoteConversion_Flag.values
    X = []
    X.append(convert_date_data(dataframe['Original_Quote_Date']))
    columns_to_skip = ['Original_Quote_Date', 'QuoteConversion_Flag',
                      'QuoteNumber']
    for c in dataframe.columns:
        if c in columns_to_skip:
            print c, 'skipped'
            continue
        if train[c].dtype == object:
            X.append(extract_object_data(dataframe[c]))
        elif c.endswith('A') or c.endswith('B'):
            X.append(extract_ordinal_data(dataframe[c]))
        else:
            X.append(extract_numeric_data(dataframe[c]))
    X = filter(lambda x: len(x) > 0, X)
    return np.vstack(X).T, y
