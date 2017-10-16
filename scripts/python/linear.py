import numpy as np
import pandas as pd
import sys
import time

from math import sqrt
from sklearn.metrics import mean_squared_log_error
from sklearn import linear_model



data = pd.read_csv('../../data/train_proc.csv',header=0)
test = pd.read_csv('../../data/test_proc.csv',header=0)

train_x=data.iloc[:-292,:80]
train_y=data.iloc[:-292,80]
test_x=data.iloc[-292:,:80]
test_y=data.iloc[-292:,80]
test_id=test.iloc[:,0]

clf = linear_model.LinearRegression()
pred = clf.fit(train_x,train_y).predict(test_x)

print(clf.score(test_x,test_y),sqrt(mean_squared_log_error(test_y, pred)))
if(len(sys.argv) >1 and sys.argv[1] == 'true'):
    prediction = clf.predict(test);
    prices=pd.DataFrame(prediction,columns=['SalePrice'])
    pred = pd.concat([test_id,prices], axis=1).to_csv('../../predictions/prediction%s.csv'%time.strftime("%c"),index=False)