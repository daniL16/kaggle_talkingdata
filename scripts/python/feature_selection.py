import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('../../data/train_outliers.csv',header=0)
test = pd.read_csv('../../data/test_proc.csv',header=0)

train_x=train.iloc[:,:train.shape[1]-1]
train_y=train.iloc[:,train.shape[1]-1]
test_id=test.iloc[:,0]


clf = GradientBoostingRegressor()
clf.fit(train_x,train_y)

model = SelectFromModel(clf,prefit=True)
mask = model.get_support()

new_features = [] # The list of your K best features
feature_names = list(train_x.columns.values)
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)

new = model.transform(train_x)
testnew = model.transform(test)
dataframe = pd.DataFrame(new, columns=new_features)

pd.concat([test_id,pd.DataFrame(testnew,columns=new_features)], axis=1).to_csv('../../data/test_fs.csv',index=False)
pd.concat([dataframe,train_y],axis=1).to_csv('../../data/train_fs.csv',index=False)

