import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import neighbors
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

data = pd.read_csv('../data/train_proc.csv',header=0)
data = data.fillna(method='ffill')

train_x=data.iloc[:-292,:37]
train_y=data.iloc[:-292,37]
test_x=data.iloc[-292:,:37]
test_y=data.iloc[-292:,37]

knn = neighbors.KNeighborsRegressor()
pred = knn.fit(train_x,train_y).predict(test_x)
kf = KFold(n_splits=10)
score = cross_val_score(knn, test_x, test_y,
cv=kf)
print(score.mean(),mean_squared_log_error(test_y, pred))
