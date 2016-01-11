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
        if data.shape[1] == 1:
            print 'constant data'
            return []
    else:
        c2i = dict((c, i) for i,c in enumerate(counts.keys(), start=1))
        if len(c2i) == 1:
            print 'constant data'
            return []
        data = np.asarray([c2i[k] for k in row])
    return data.T

def extract_numeric_data(row, normalize_numeric = True):
    row = row.fillna(row.median())
    if len(row.unique()) == 1:
        return []

    if normalize_numeric:
        row = row - row.mean()
        row = row/row.std()
    return row.values

def extract_ordinal_data(row, normalize_numeric = False):
    return extract_numeric_data(row, normalize_numeric)


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
            X.append(extract_object_data(dataframe[c], cat2vectors))
        elif c.endswith('A') or c.endswith('B'):
            X.append(extract_ordinal_data(dataframe[c]))
        else:
            X.append(extract_numeric_data(dataframe[c], normalize_numeric))
    X = filter(lambda x: len(x) > 0, X)
    return np.vstack(X).T, y

def load_data(get_nan_feature = True):
    train = pd.read_csv('train.csv')
    train.replace([' ', '', -1], np.nan, inplace = True)
    if get_nan_feature:
        train['NaNCount'] = train.isnull().sum(axis=1)
    X, y = convert_data(train, cat2vectors = False, normalize_numeric = False)
    print X.shape, y.shape
    return X, y






