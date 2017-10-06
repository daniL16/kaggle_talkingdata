import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

data = pd.read_csv('../data/train_proc.csv',header=0)

train_x=data.iloc[:-292,:37]
train_y=data.iloc[:-292,37]
test_x=data.iloc[-292:,:37]
test_y=data.iloc[-292:,37]

clf = ensemble.AdaBoostRegressor()
pred = clf.fit(train_x,train_y).predict(test_x)

kf = KFold(n_splits=10)
score = cross_val_score(clf,test_x, test_y,
cv=kf)

print(clf.score(test_x,test_y),mean_squared_log_error(test_y, pred))