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
from rusboost import RUSBoost
from sklearn import tree

from imblearn.under_sampling import RandomUnderSampler,AllKNN,ClusterCentroids



from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

path='/media/dani/E892136C92133E8E/TFG/data/'
#train = pd.read_csv(path+"train.csv",parse_dates=True,nrows=int(sys.argv[2]))

'''
df1= pd.read_csv('/media/dani/E892136C92133E8E/TFG/predictions/group75ok.csv')
df2=pd.read_csv('/media/dani/E892136C92133E8E/TFG/predictions/grouping75old.csv')
df = pd.DataFrame()
df['click_id'] = df2['click_id']
df['is_attributed'] = df1
df.to_csv('/media/dani/E892136C92133E8E/TFG/predictions/group75ok2.csv',index=False)
'''

#funcion auxiliar para monitorizar el uso de memoria
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


#funciones para optimizar los conjuntos de train/test
def low_ram_train_read(nrows):
    
    features = ['ip', 'app', 'device', 'os', 'channel', 'hour','day','year','month','click_time_timestamp', 'is_attributed']
    int_features = ['ip', 'app', 'device', 'os', 'channel', 'hour','day','year','month','click_time_timestamp']
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

    features = ['ip', 'app', 'device', 'os', 'channel', 'hour','day','year','month','click_time_timestamp']
    int_features = ['ip', 'app', 'device', 'os', 'channel', 'hour','day','year','month','click_time_timestamp']

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
    if type='random':
        und = RandomUnderSampler(ratio='majority',random_state=42)
    else if type='knn':
        und = AllKNN(ratio='majority',random_state=42,njobs=4)
    else if type='centroids':
        und = ClusterCentroids(ratio='majority',njobs=4)
    x, y= und.fit_sample(train,y)
    x = pd.DataFrame(x,columns=train.columns.values)
    y = pd.DataFrame(y,columns=['is_attributed'])
    return x,y
    
    
#pollas en ollas


def modelfit(alg, dtrain,y, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
   
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], y,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(y ,dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))
                    
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')


train = low_ram_train_read(float(sys.argv[1]))
print('Train : '+mem_usage(train))

y = train['is_attributed']
train.drop(['is_attributed'], axis=1, inplace=True)

x,y =undersampling(sys.argv[2])
print('undersampling, nrows: '+len(x))
print('Train : '+mem_usage(train))

del train

most_freq_hours_in_data    = [4, 5, 9, 10, 13, 14]
middle1_freq_hours_in_data = [16, 17, 22]
least_freq_hours_in_data   = [6, 11, 15]
x['in_hh'] = (   4 
                     - 3*x['hour'].isin(  most_freq_hours_in_data ) 
                     - 2*x['hour'].isin(  middle1_freq_hours_in_data ) 
                     - 1*x['hour'].isin( least_freq_hours_in_data ) ).astype('uint8')

gp    = x[['ip', 'day', 'in_hh', 'channel']].groupby(by=['ip', 'day', 'in_hh'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_hh'})
x = x.merge(gp, on=['ip','day','in_hh'], how='left')
x.drop(['in_hh'], axis=1, inplace=True)

x['nip_day_hh'] = x['nip_day_hh'].astype('uint16')
del gp
gc.collect()

print('Train : '+mem_usage(x))


params = {'eta': 0.1, 
          'max_depth': 4, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':100,
          'alpha':4,
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'random_state': 99, 
          'silent': False}
                 
x1, x2, y1, y2 = train_test_split(x, y, test_size=0.1, random_state=99)
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 360, watchlist, maximize=True, verbose_eval=10)
del x1,x2,y1,y2,watchlist
gc.collect()
print(model.feature_importances_)

print("making predictions")
test = low_ram_test_read()
sub = pd.DataFrame()
most_freq_hours_in_data    = [4, 5, 9, 10, 13, 14]
middle1_freq_hours_in_data = [16, 17, 22]
least_freq_hours_in_data   = [6, 11, 15]
test['in_hh'] = (   4 
                 - 3*test['hour'].isin(  most_freq_hours_in_data ) 
                 - 2*test['hour'].isin(  middle1_freq_hours_in_data ) 
                 - 1*test['hour'].isin( least_freq_hours_in_data ) ).astype('uint8')

gp    = test[['ip', 'day', 'in_hh', 'channel']].groupby(by=['ip', 'day', 'in_hh'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_hh'})
test = test.merge(gp, on=['ip','day','in_hh'], how='left')
test.drop(['in_hh'], axis=1, inplace=True)
test['nip_day_hh'] = test['nip_day_hh'].astype('uint32')
del gp
gc.collect()
sub['click_id'] = test['click_id']
test.drop(['click_id'], axis=1, inplace=True)
  
sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
if(len(sys.argv) > 3) :
    sub.to_csv(path+'../predictions/' + sys.argv[3],index=False)
else:
    sub.to_csv('./predictions/prediction%s.csv'%time.strftime("%c"),index=False)
