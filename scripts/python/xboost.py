import numpy as np
import pandas as pd
import sys
import time
import xgboost as xgb
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score,KFold
from math import sqrt
from sklearn.metrics import mean_squared_log_error,make_scorer
from sklearn.preprocessing import Normalizer
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('../../data/train_fs.csv',header=0,index_col='Id')
test = pd.read_csv('../../data/test_fs.csv',header=0,index_col='Id')

train_y = train.pop('SalePrice')
train_x = train

xgb = xgb.XGBRegressor().fit(train_x,train_y)

kf = KFold(len(train_x), n_folds=7, random_state=42,shuffle=True)
score= np.sqrt(-cross_val_score(xgb, train.values, train_y, scoring="neg_mean_squared_error", cv = kf))
print((score.mean()))
if(len(sys.argv) >1 and sys.argv[1] == 'true'):
    prediction = xgb.predict(test);
    prediction = np.exp(prediction)
    prices=pd.DataFrame(prediction,columns=['SalePrice'],index=test.index).to_csv('../../predictions/prediction%s.csv'%time.strftime("%c"))
  