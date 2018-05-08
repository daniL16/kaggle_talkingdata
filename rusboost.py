from sklearn import svm
from sklearn import tree
from math import log
import random
import numpy as np
import pandas as pd
import imblearn
import sys
from imblearn.under_sampling import RandomUnderSampler


class AdaBoost:
    def __init__(self, M, depth):
        self.M = M
        self.depth = depth
        self.undersampler = RandomUnderSampler(return_indices=True,replacement=False)


    def fit(self, X, Y):
        self.models = []
        self.alphas = []

        N, _ = X.shape
        W = np.ones(N) / N

        for m in range(self.M):
            print(m)
            tr = tree.DecisionTreeClassifier(max_depth=self.depth, splitter='best')

            X_undersampled, y_undersampled, chosen_indices = self.undersampler.fit_sample(X, Y)

            tr.fit(X_undersampled, y_undersampled, sample_weight=W[chosen_indices])

            P = tr.predict(X)

            err = np.sum(W[P != Y])

            if err > 0.5:
                m = m - 1
            if err <= 0:
                err = 0.0000001
            else:
                try:
                    if (np.log(1 - err) - np.log(err)) == 0 :
                        alpha = 0
                    else:
                        alpha = 0.5 * (np.log(1 - err) - np.log(err))
                    W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1
                except:
                    alpha = 0
                    # W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1

                self.models.append(tr)
                self.alphas.append(alpha)
                W = W.values
    def predict(self, X):
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tr in zip(self.alphas, self.models):
            FX += alpha * tr.predict(X)
      
        return np.sign(FX)

    def predict_proba(self, X):
        proba = sum(tr.predict_proba(X) * alpha for tr , alpha in zip(self.models,self.alphas) )


        proba = np.array(proba)


        proba = proba / sum(self.alphas)

        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        # proba =  np.linspace(proba)
        # proba = np.array(proba).astype(float)
        proba = proba /  normalizer

        return proba[:,0]

    
def low_ram_train_read(nrows,init_row=0):
    
    features = ['ip', 'app', 'device', 'os', 'channel', 'hour','click_time_timestamp','qty','ip_app_count','ip_app_os_count','is_attributed']
   

    int_features = ['ip', 'app', 'device', 'os', 'channel', 'hour','click_time_timestamp', 'qty','ip_app_count','ip_app_os_count']
    bool_features = ['is_attributed']

    for feature in features:
        print("Loading ", feature)
        #Import data one column at a time
        if init_row == 0 :
            train_unit = pd.read_csv(path+'train_proc.csv', usecols=[feature],nrows=nrows) #Change this from "train_sample" to "train" when you're comfortable
        else:
            train_unit = pd.read_csv(path+'train_proc.csv', usecols=[feature],nrows=nrows,skiprows=[1,int(init_row)]) #Change this from "train_sample" to "train" when you're comfortable!
    
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



path='/media/dani/E892136C92133E8E/TFG/data/'

train = low_ram_train_read(int(sys.argv[1]))
y = train['is_attributed']
train.drop(['is_attributed'], axis=1, inplace=True)

if sys.argv[3]=='ada':
    model = AdaBoost(250,250)
    model.fit(train,y)
else:
    base_classifier = tree.DecisionTreeClassifier()
    model = RUSBoost(train,y,base_classifier,10,0.05)
    model.learning()

del train
print("making predictions")

test = low_ram_test_read()
sub = pd.DataFrame()
proba = pd.DataFrame()
sub['click_id'] = test['click_id']
proba['click_id'] = test['click_id']
test.drop(['click_id'], axis=1, inplace=True)

if sys.argv[3]=='ada':
    sub['is_attributed'] = model.predict(test)
    proba['is_attributed'] = model.predict_proba(test)
    
else:
    sub['is_attributed'] = model.classify(test)

if(len(sys.argv) > 2) :
    sub.to_csv(path+'../predictions/' + sys.argv[2],index=False)
    sub.to_csv(path+'../predictions/proba_' + sys.argv[2],index=False)
else:
    sub.to_csv('./predictions/prediction%s.csv'%time.strftime("%c"),index=False)

