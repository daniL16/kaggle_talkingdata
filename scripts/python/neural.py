from sklearn.neural_network import MLPRegressor
import pandas as pd
import sys
import time
from math import sqrt
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error


data = pd.read_csv('../../data/train_proc.csv',header=0)
test = pd.read_csv('../../data/test_proc.csv',header=0)

scaler = StandardScaler() 
 
train_x=data.iloc[:-292,:80]
train_y=data.iloc[:-292,80]
test_x=data.iloc[-292:,:80]
test_y=data.iloc[-292:,80]
test_id=test.iloc[:,0]

scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

clf = MLPRegressor(max_iter=500000)
pred = clf.fit(train_x,train_y).predict(test_x)

#print(clf.score(test_x,test_y),sqrt(mean_squared_log_error(test_y, pred)),sqrt(mean_squared_error(test_y, pred)))
if(len(sys.argv) >1 and sys.argv[1] == 'true'):
    prediction = clf.predict(test);
    prices=pd.DataFrame(prediction,columns=['SalePrice'])
    pred = pd.concat([test_id,prices], axis=1).to_csv('../../predictions/prediction%s.csv'%time.strftime("%c"),index=False)