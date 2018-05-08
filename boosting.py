#usage : python3 boosting.py nrows method outputfile

import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
import sys
import time
import gc
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import tree

from imblearn.under_sampling import RandomUnderSampler,AllKNN,ClusterCentroids



from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

path='/media/dani/E892136C92133E8E/TFG/data/'
#train = pd.read_csv(path+"train.csv",parse_dates=True,nrows=int(sys.argv[2]))

#funcion auxiliar para monitorizar el uso de memoria
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


#funciones para optimizar los conjuntos de train/test
def low_ram_train_read(nrows,init_row=0):
    
    features = ['ip', 'app', 'device', 'os', 'channel', 'hour','click_time_timestamp','qty','ip_app_count','ip_app_os_count','is_attributed']
    int_features = ['ip', 'app', 'device', 'os', 'channel', 'hour','click_time_timestamp', 'qty','ip_app_count','ip_app_os_count']
    bool_features = ['is_attributed']
    for feature in features:
        print("Loading ", feature)
        #Import data one column at a time
        train_unit = pd.read_csv(path+'train_proc.csv', usecols=[feature],nrows=nrows) #Change this from "train_sample" to "train" when you're comfortable!
    
        #Pandas imports the numeric data as int64...the following should downgrade that to uint16, saving ~1GB in RAM for each column
        if feature in int_features:    train_unit = pd.to_numeric(train_unit[feature], downcast='unsigned')
        #Convert time data to datetime data, instead of strings
        
        #Converts the target variable from int64 to boolean. Can also get away with uint16.
        elif feature in bool_features: train_unit = train_unit[feature].astype('bool')
    
        #Make and append each column's data to a dataframe.
        if feature == 'ip': train = pd.DataFrame(train_unit)
        else: train[feature] = train_unit
    return train


def low_ram_test_read():

    features = ['click_id','ip', 'app', 'device', 'os', 'channel', 'hour','click_time_timestamp','qty','ip_app_count','ip_app_os_count']
    int_features = ['ip', 'app', 'device', 'os', 'channel', 'hour','click_time_timestamp','qty','ip_app_count','ip_app_os_count']
    time_features=[]
    for feature in features:
        print("Loading ", feature)
        #Import data one column at a time
        test_unit = pd.read_csv(path+'test_proc.csv', usecols=[feature]) #Change this from "train_sample" to "train" when you're comfortable!
    
        #Pandas imports the numeric data as int64...the following should downgrade that to uint16, saving ~1GB in RAM for each column
        if feature in int_features:    
            test_unit = pd.to_numeric(test_unit[feature], downcast='unsigned')
        #Convert time data to datetime data, instead of strings
        elif feature in time_features: test_unit=pd.to_datetime(test_unit[feature])
       
        #Make and append each column's data to a dataframe.
        if feature == 'click_id': test = pd.DataFrame(test_unit)
        else: test[feature] = test_unit
    return test



#undersampling
def undersampling(type):
    if type=='random':
        und = RandomUnderSampler(ratio='majority',random_state=42)
    elif type=='knn':
        und = AllKNN(ratio='majority',random_state=42,n_jobs=4)
    elif type=='centroids':
        und = ClusterCentroids(ratio='majority',n_jobs=-1)
    x,y= und.fit_sample(train,label)
    x = pd.DataFrame(x,columns=train.columns.values)
    y = pd.DataFrame(y,columns=['is_attributed'])
    
    return x,y
    
step = 300000
iters = int(float(sys.argv[1])/step)
x = pd.DataFrame()
y= pd.DataFrame()
for i in range(0,iters):
    init_row=i*step
    train = low_ram_train_read(step,init_row)
    label = train['is_attributed']
    #print(label[label==True])
    train.drop(['is_attributed'], axis=1, inplace=True)
    xx,yy =undersampling(sys.argv[2])
    del train
    x.append(xx)
    y.append(yy)
    print(len(x))
    
'''
t1 = int(float(sys.argv[1])/2)
train = low_ram_train_read(t1)
print('Train : '+mem_usage(train))

label = train['is_attributed']
train.drop(['is_attributed'], axis=1, inplace=True)

x1,y1 =undersampling(sys.argv[2])

del train

train = low_ram_train_read(t1,t1)
label = train['is_attributed']
train.drop(['is_attributed'], axis=1, inplace=True)

x2,y2 =undersampling(sys.argv[2])
del train 

x = pd.concat([x1,x2])
y = pd.concat([y1,y2])

del x1,x2,y1,y2

print('Train : '+mem_usage(x))
'''



y = x['is_attributed']
x.drop(['is_attributed'], axis=1, inplace=True)

#params={'colsample_bylevel': 1.0, 'colsample_bytree': 0.9, 'max_depth': 5, 'min_child_weight': 0, 'n_estimators': 250, 'subsample': 1.0}
params = {'eta': 0.3,
          'tree_method': "gpu_exact",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 99, 
          'silent': True}
params2 = {'eta': 0.07,
          'lambda': 21.0033,
          'max_delta_step' : 5.0876,
          'scale_pos_weight' : 150,
           'tree_method' : 'exact',
         'nrounds' : 2000,
          'max_depth': 4, 
          'subsample': 1.0, 
          'colsample_bytree': 0.5962, 
          'colsample_bylevel':0.5214,
          'min_child_weight':96,
          'alpha':0,
           'gamma' : 6.1142,
          'objective': 'binary:logistic', 
          'scale_pos_weight':150,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 99, 
          'silent': True,
           'booster' : "gbtree"}
x1, x2, y1, y2 = train_test_split(x, y, test_size=0.1, random_state=99)
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train( params2 , xgb.DMatrix(x1, y1), 250, watchlist, maximize=True, verbose_eval=10)
del x1,x2,y1,y2,watchlist
gc.collect()

print(model.get_fscore())
print("making predictions")
test = low_ram_test_read()
sub = pd.DataFrame()

sub['click_id'] = test['click_id']
test.drop(['click_id'], axis=1, inplace=True)
#test.drop(['day','wday'], axis=1, inplace=True)
#test.drop(['next_click'], axis=1, inplace=True)

sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)

if(len(sys.argv) > 3) :
    sub.to_csv(path+'../predictions/' + sys.argv[3],index=False)
else:
    sub.to_csv('./predictions/prediction%s.csv'%time.strftime("%c"),index=False)
