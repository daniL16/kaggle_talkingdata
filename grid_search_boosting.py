import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
import sys
import time
import gc
import os, psutil, numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold

path='/media/dani/E892136C92133E8E/TFG/data/'
#train = pd.read_csv(path+"train.csv",parse_dates=True,nrows=int(sys.argv[2]))

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

#target 184903891
def low_ram_train_read(nrows):
    features = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
    int_features = ['ip', 'app', 'device', 'os', 'channel']
    time_features = ['click_time', 'attributed_time']
    bool_features = ['is_attributed']

    for feature in features:
        print("Loading ", feature)
        #Import data one column at a time
        train_unit = pd.read_csv(path+'train.csv', usecols=[feature],nrows=nrows) #Change this from "train_sample" to "train" when you're comfortable!
    
        #Pandas imports the numeric data as int64...the following should downgrade that to uint16, saving ~1GB in RAM for each column
        if feature in int_features:    train_unit = pd.to_numeric(train_unit[feature], downcast='unsigned')
        #Convert time data to datetime data, instead of strings
        elif feature in time_features: train_unit=pd.to_datetime(train_unit[feature])
        #Converts the target variable from int64 to boolean. Can also get away with uint16.
        elif feature in bool_features: train_unit = train_unit[feature].astype('bool')
    
        #Make and append each column's data to a dataframe.
        if feature == 'ip': train = pd.DataFrame(train_unit)
        else: train[feature] = train_unit
    return train

def low_ram_test_read():

    features = ['click_id','ip', 'app', 'device', 'os', 'channel', 'click_time']
    int_features = ['ip', 'app', 'device', 'os', 'channel']
    time_features = ['click_time']

    for feature in features:
        print("Loading ", feature)
        #Import data one column at a time
        test_unit = pd.read_csv(path+'test.csv', usecols=[feature]) #Change this from "train_sample" to "train" when you're comfortable!
    
        #Pandas imports the numeric data as int64...the following should downgrade that to uint16, saving ~1GB in RAM for each column
        if feature in int_features:    
            test_unit = pd.to_numeric(test_unit[feature], downcast='unsigned')
        #Convert time data to datetime data, instead of strings
        elif feature in time_features: test_unit=pd.to_datetime(test_unit[feature])
       
        #Make and append each column's data to a dataframe.
        if feature == 'click_id': test = pd.DataFrame(test_unit)
        else: test[feature] = test_unit
    return test

def processDates(df):
    df['click_time'] = pd.to_datetime(df['click_time']).dt.date
    df['click_time_year'] =pd.to_numeric(df['click_time'].map(lambda x: x.year),downcast='unsigned')
    df['click_time_month'] =pd.to_numeric(df['click_time'].map(lambda x: x.month),downcast='unsigned')
    df['click_time_day'] =pd.to_numeric(df['click_time'].map(lambda x: x.day),downcast='unsigned')
    return df


train = low_ram_train_read(int(sys.argv[2]))
train = processDates(train)
y = train['is_attributed'].values
train.drop(['is_attributed', 'attributed_time','click_time'], axis=1, inplace=True)

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1001)

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}

#x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=99)
#watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

#model = xgb.train(params, xgb.DMatrix(x1, y1), 260, watchlist, maximize=True, verbose_eval=10
#del x1,x2,y1,y2,watchlist

xgb_model = xgb.XGBClassifier()
clf = GridSearchCV(xgb_model, parameters, n_jobs=4, 
                   cv=skf.split(train,y), 
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(train,y)
del train,y
gc.collect()

if(len(sys.argv) >1 and sys.argv[1] == 'true'):
    print("making predictions")
    test = low_ram_test_read()
    test = processDates(test)
    sub = pd.DataFrame()
    sub['click_id'] = test['click_id']
    test.drop(['click_id','click_time'], axis=1, inplace=True)
    
    sub['is_attributed'] = clf.best_estimator_.predict_proba(test)[:,1]
    #sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
    if(len(sys.argv) > 3) :
        sub.to_csv('./predictions/' + sys.argv[3],index=False)
    else:
        sub.to_csv('./predictions/prediction%s.csv'%time.strftime("%c"),index=False)