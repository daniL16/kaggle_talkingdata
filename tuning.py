from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib
from matplotlib import pyplot


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


# load data

data = low_ram_train_read(sys.argv[1])
dataset = data.values
# split data into X and y
X = dataset[:,0:94]
y = dataset[:,94]
# encode string class values as integers
# grid search
model = XGBClassifier()
n_estimators = range(50, 400, 50)
param_grid = dict(n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X,y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%