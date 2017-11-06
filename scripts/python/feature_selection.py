import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('../../data/train_proc.csv',header=0,index_col='Id')
test = pd.read_csv('../../data/test_proc.csv',header=0,index_col='Id')

train_x=train.iloc[:,:train.shape[1]-1]
train_y=train.iloc[:,train.shape[1]-1]

clf = GradientBoostingRegressor()
clf.fit(train_x,train_y)

model = SelectFromModel(clf,prefit=True)
mask = model.get_support()
new_features = [] # The list of your K best features
feature_names = list(train_x.columns.values)
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)

train_new = model.transform(train_x)
test_new = model.transform(test)

train = pd.concat([pd.DataFrame(train_new, columns=new_features,index=train.index),train_y],axis=1)
test = pd.DataFrame(test_new,columns=new_features,index=test.index)

test.to_csv('../../data/test_fs.csv')
train.to_csv('../../data/train_fs.csv')

