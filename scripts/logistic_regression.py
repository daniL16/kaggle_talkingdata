import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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

clf = LogisticRegression(solver='sag', max_iter=100, random_state=42)
kf = KFold(len(train_x), n_folds=10, random_state=42)
score = cross_val_score(clf, train_x, train_y,
cv=kf)
print(score.mean())