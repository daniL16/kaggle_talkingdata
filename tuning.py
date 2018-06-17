import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler, AllKNN, ClusterCentroids

import sys
import tools as Tools

train = Tools.low_ram_train_read(int(sys.argv[1]))
is_attributed = data['is_attributed']
train.drop(['is_attributed'],axis=1)

und = RandomUnderSampler(ratio='majority',random_state=42)
x,y= und.fit_sample(train,is_attributed)
x = pd.DataFrame(x,columns=train.columns.values)

del train

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

