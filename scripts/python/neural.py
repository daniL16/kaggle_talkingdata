import pandas as pd
import sys
import time

from math import sqrt
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor


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

reg = MLPRegressor(max_iter=3000)
pred = reg.fit(train_x,train_y).predict(test_x)

print(reg.score(test_x,test_y),sqrt(mean_squared_log_error(test_y, pred)))
if(len(sys.argv) >1 and sys.argv[1] == 'true'):
    prediction = reg.predict(test);
    prices=pd.DataFrame(prediction,columns=['SalePrice'])
    pred = pd.concat([test_id,prices], axis=1).to_csv('../../predictions/prediction%s.csv'%time.strftime("%c"),index=False)