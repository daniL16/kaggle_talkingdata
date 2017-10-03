import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import neighbors
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

knn = neighbors.KNeighborsRegressor(5)
pred = knn.fit(train_x,train_y).predict(test_x)

kf = KFold(n_splits=10)
score = cross_val_score(knn, train_x, train_y,
cv=kf)
print(score.mean())