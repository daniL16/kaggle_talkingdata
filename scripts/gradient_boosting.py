import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

data = pd.read_csv('../data/train.csv',header=0)
data = data._get_numeric_data()
data.isnull().any()
data = data.fillna(method='ffill')

train_x=data.iloc[:-20,:37]
train_y=data.iloc[:-20,37]
test_x=data.iloc[-20:,:37]
test_y=data.iloc[-20:,37]

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

kf = KFold(n_splits=10)
score = cross_val_score(clf, train_x, train_y,
cv=kf)
print(score.mean())