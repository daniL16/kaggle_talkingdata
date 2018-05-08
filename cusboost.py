from sklearn import svm
from sklearn import tree
from math import log
import random
import numpy as np
import pandas as pd
import imblearn
import sys
from imblearn.under_sampling import RandomUnderSampler,ClusterCentroids

class CusBoost:
    def __init__(self,instances,labels,k):
        self.weight = []
        self.X= instances
        self.Y = labels
        self.k = k
        self.init_w = 1.0/len(self.X)
        for i in range(len(self.X)):
            self.weight.append(self.init_w)
    def learning(self):
        self.models = []
        self.alphas = []

        N, _ = self.X.shape
        W = np.ones(N) / N
        for i in range(self.k):
            print(i)
            cus = ClusterCentroids(ratio='majority')
            x_undersampled,y_undersampled= cus.fit_sample(self.X,self.Y)
            cl = tree.DecisionTreeClassifier( splitter='best')
            cl.fit(x_undersampled, y_undersampled)

            P = cl.predict(self.X)
            
            err = np.sum(W[P != self.Y])

            if err > 0.5:
                i = i - 1
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

                self.models.append(cl)
                self.alphas.append(alpha)
                

    def predict(self,X):
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tr in zip(self.alphas, self.models):
            FX += alpha * tr.predict(X)
      
        return np.sign(FX)
            
            
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


model = CusBoost(train,y,250)
model.learning()

del train
print("making predictions")

test = low_ram_test_read()
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop(['click_id'], axis=1, inplace=True)


sub['is_attributed'] = model.predict(test)


if(len(sys.argv) > 2) :
    sub.to_csv(path+'../predictions/' + sys.argv[2],index=False)
else:
    sub.to_csv('./predictions/prediction%s.csv'%time.strftime("%c"),index=False)
     
            
          