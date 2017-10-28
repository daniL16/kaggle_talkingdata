import numpy as np
import pandas as pd
import sys
import time
import xgboost as xgb
from sklearn.cross_validation import cross_val_score,KFold
from math import sqrt
from sklearn.metrics import mean_squared_log_error,make_scorer
from sklearn.preprocessing import Normalizer
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../../data/train_outliers.csv',header=0,index_col='Id')
test = pd.read_csv('../../data/test_proc.csv',header=0,index_col='Id')
#norm = Normalizer()
#test = pd.DataFrame(norm.fit_transform(test),columns=test.columns.values,index = test.index)
#train_x = pd.DataFrame(norm.fit_transform(train_x),columns=train_x.columns.values,index=train.index)
#train["SalePrice"] = np.log1p(train["SalePrice"])

train_y = train.pop('SalePrice')
train_x = train
del (train)


xgb = xgb.XGBRegressor().fit(train_x,train_y)

kf = KFold(len(train_x), n_folds=10, random_state=42)
score = cross_val_score(xgb, train_x, train_y,cv=kf, scoring=make_scorer(mean_squared_log_error))

print(sqrt(score.mean()))
if(len(sys.argv) >1 and sys.argv[1] == 'true'):
    prediction = xgb.predict(test);
    prices=pd.DataFrame(prediction,columns=['SalePrice'],index=test.index).to_csv('../../predictions/prediction%s.csv'%time.strftime("%c"))
  