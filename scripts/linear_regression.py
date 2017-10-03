import numpy as np
import pandas as pd
from sklearn import linear_model

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

lm = linear_model.LinearRegression()

kf = KFold(n_splits=10)
score = cross_val_score(lm, train_x, train_y,cv=kf)
print(score.mean())