# usage : python3 boosting.py nrows [outputfile] [under_sampling method]

import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import sys
import time
import gc

from rusboost import RUSBoost
from cusboost import CUSBoost

from tools import Tools

# variables globales
step = 300000
iters = int(float(sys.argv[1]) / step)
x = pd.DataFrame()
y = pd.DataFrame()

# undersampling
def undersampling(type):
    if type == 'random':
        und = RandomUnderSampler(ratio='majority', random_state=42)
    elif type == 'knn':
        und = AllKNN(ratio='majority', random_state=42, n_jobs=4)
    elif type == 'centroids':
        und = ClusterCentroids(ratio='majority', n_jobs=-1)
    x, y = und.fit_sample(train, label)
    x = pd.DataFrame(x, columns=train.columns.values)
    y = pd.DataFrame(y, columns=['is_attributed'])

    return x, y





if (len(sys.argv) > 3):
    # Aplicamos RUSBoosting
    if sys.argv[3] == 'rus':
        rus = RUSBoost(250, 4, True)
        x = Tools.low_ram_train_read(int(sys.argv[1]))
        y = x['is_attributed']
        x.drop(['is_attributed'], axis=1, inplace=True)
        rus.fit(x, y)
        del x, y
        gc.collect
        print("making predictions")
        test = Tools.low_ram_test_read()
        sub = pd.DataFrame()
        sub['click_id'] = test['click_id']
        test.drop(['click_id'], axis=1, inplace=True)
        sub['is_attributed'] = rus.predict(test)
        sub.to_csv(path + '../predictions/' + sys.argv[2], index=False)
        sys.exit(0)
       
    #Aplicamos CUSBoosting
    elif sys.argv[3] == 'cus':
            rus = CUSBoost(250,4,True)
            x = Tools.low_ram_train_read(int(sys.argv[1]))
            y = x['is_attributed']
            x.drop(['is_attributed'], axis=1, inplace=True)
            rus.fit(x,y)
            del x,y
            gc.collect
            print("making predictions")
            test = Tools.low_ram_test_read()
            sub = pd.DataFrame()
            sub['click_id'] = test['click_id']
            test.drop(['click_id'], axis=1, inplace=True)
            sub['is_attributed'] = rus.predict(test)
            sub.to_csv(path+'../predictions/' + sys.argv[2],index=False)
            sys.exit(0)
    #boosting con un conjunto balanceado
    #xgboost con un conjunto balanceado
    else:
        for i in range(0, iters):
            init_row = i * step
            train = Tools.low_ram_train_read(step, init_row)
            label = train['is_attributed']
            train.drop(['is_attributed'], axis=1, inplace=True)
            xx, yy = undersampling(sys.argv[3])
            del train
            x.append(xx)
            y.append(yy)
#boosting
else:
    x = Tools.low_ram_train_read(int(sys.argv[1]))
    y = x['is_attributed']
    x.drop(['is_attributed'], axis=1, inplace=True)
    print(Tools.mem_usage(x))

    
params2 = {'eta': 0.07,
           'lambda': 21.0033,
           'max_delta_step': 5.0876,
           'scale_pos_weight': 150,
           'tree_method': 'exact',
           'nrounds': 2000,
           'max_depth': 4,
           'subsample': 1.0,
           'colsample_bytree': 0.5962,
           'colsample_bylevel': 0.5214,
           'min_child_weight': 96,
           'alpha': 0,
           'gamma': 6.1142,
           'objective': 'binary:logistic',
           'scale_pos_weight': 150,
           'eval_metric': 'auc',
           'nthread': 8,
           'random_state': 99,
           'silent': True,
           'booster': "gbtree"}

x1, x2, y1, y2 = train_test_split(x, y, test_size=0.1, random_state=99)
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params2, xgb.DMatrix(x1, y1), 250, watchlist, maximize=True, verbose_eval=10)
del x, y, x1, x2, y1, y2, watchlist
gc.collect()
print(model.get_fscore())
print("making predictions")
test = Tools.low_ram_test_read()
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop(['click_id'], axis=1, inplace=True)
sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)

if (len(sys.argv) > 2):
    sub.to_csv(path + '../predictions/' + sys.argv[2], index=False)
else:
    sub.to_csv(path + '../predictions/prediction%s.csv' % time.strftime("%c"), index=False)
