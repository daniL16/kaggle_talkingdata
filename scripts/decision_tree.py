import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

data = pd.read_csv('../data/train.csv',header=0)
data = data._get_numeric_data()
data.isnull().any()
data = data.fillna(method='ffill')

train_x=data.iloc[:-20,:37]
train_y=data.iloc[:-20,37]
test_x=data.iloc[-20:,:37]
test_y=data.iloc[-20:,37]

regr = DecisionTreeRegressor(max_depth=5)

kf = KFold(len(train_x), n_folds=10, random_state=42)
score = cross_val_score(regr, train_x, train_y,
cv=kf)
print(score.mean())