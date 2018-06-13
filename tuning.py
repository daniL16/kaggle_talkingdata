import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import sys

path='/media/dani/E892136C92133E8E/TFG/data/'


def low_ram_train_read(nrows):
    
    features = ['ip', 'app', 'device', 'os', 'channel', 'hour','day','nextClick','click_time_timestamp', 'is_attributed']
    int_features = ['ip', 'app', 'device', 'os', 'channel', 'hour','nextClick','month','click_time_timestamp']
    bool_features = ['is_attributed']
    for feature in features:
        train_unit = pd.read_csv(path + 'train_proc.csv', usecols=[feature], nrows=nrows)
    
        if feature in int_features:    train_unit = pd.to_numeric(train_unit[feature], downcast='unsigned')
        
        elif feature in bool_features: train_unit = train_unit[feature].astype('bool')

        if feature == 'ip': train = pd.DataFrame(train_unit)
        else: train[feature] = train_unit
    return train


def low_ram_test_read():

    features = ['click_id','ip', 'app', 'device', 'os', 'channel', 'hour','day','nextClick','click_time_timestamp']
    int_features = ['ip', 'app', 'device', 'os', 'channel', 'hour','day','nextClick','click_time_timestamp']

    for feature in features:

        test_unit = pd.read_csv(path + 'test_proc.csv', usecols=[feature])
        if feature in int_features:    
            test_unit = pd.to_numeric(test_unit[feature], downcast='unsigned')

        elif feature in time_features: test_unit=pd.to_datetime(test_unit[feature])
       
        if feature == 'click_id': test = pd.DataFrame(test_unit)
        else: test[feature] = test_unit
    return test


data = low_ram_train_read(int(sys.argv[1]))
label = data['is_attributed']
data.drop(['is_attributed'],axis=1)

und = RandomUnderSampler(ratio='majority',random_state=42)
x,y= und.fit_sample(data,label)
x = pd.DataFrame(x,columns=data.columns.values)


del data

model = XGBClassifier()
n_estimators = range(250, 350, 50)
max_depth = range(5,16,2)
subsample = [0.8,0.9,1.0]
#min_samples_split = range(200,1001,200)
colsample_bytree=[0.5,0.6,0.7,0.8,0.9,1.0]
#min_samples_leaf = range(30,71,10)
min_child_weight =range(0,150,20)
colsample_bylevel=[0.5,0.6,0.7,0.8,0.9,1.0]
param_grid = dict(max_depth=max_depth,n_estimators=n_estimators,subsample=subsample,colsample_bytree=colsample_bytree,min_child_weight=min_child_weight,colsample_bylevel=colsample_bylevel)
kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(x, y)
print("Score %f , parametros %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
