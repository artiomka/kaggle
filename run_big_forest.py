import numpy as np
from sklearn.ensemble import RandomForestClassifier
X, y = np.load('train_X.npy'), np.load('train_y.npy')
print X.shape, y.shape
clf = RandomForestClassifier(max_features = 200, n_estimators = 15000, n_jobs = -1)
clf.fit(X, y)
print 'training done'
print clf.score(X, y)

from sklearn.externals import joblib
joblib.dump(clf, 'big_forest.pkl')
