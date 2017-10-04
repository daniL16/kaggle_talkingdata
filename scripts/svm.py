import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

data = pd.read_csv('../data/train.csv',header=0)

train_x=data.iloc[:-292,:37]
train_y=data.iloc[:-292,37]
test_x=data.iloc[-292:,:37]
test_y=data.iloc[-292:,37]

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
pred = svr_poly.fit(train_x,train_y).predict(test_x)
kf = KFold(n_splits=10)
score = cross_val_score(svr_poly, train_x, train_y,
cv=kf)
print(score.mean(),mean_squared_log_error(test_y, pred))