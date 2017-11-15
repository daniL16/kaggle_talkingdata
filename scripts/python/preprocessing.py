import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy import stats
from sklearn.preprocessing import OneHotEncoder,Imputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../../data/train.csv',header=0,index_col='Id')
train_y = train.pop('SalePrice')
test = pd.read_csv('../../data/test.csv',header=0,index_col='Id')

#NaN
train['Alley'] = train['Alley'].fillna('None')
test['Alley'] = test['Alley'].fillna('None')
train['BsmtQual'] = train['BsmtQual'].fillna('None')
test['BsmtQual'] = test['BsmtQual'].fillna('None')
train['BsmtCond'] = train['BsmtCond'].fillna('None')
test['BsmtCond'] = test['BsmtCond'].fillna('None')
train['BsmtExposure'] = train['BsmtExposure'].fillna('None')
test['BsmtExposure'] = test['BsmtExposure'].fillna('None')
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('None')
test['BsmtFinType1'] = test['BsmtFinType1'].fillna('None')
train['BsmtFinType2'] = train['BsmtFinType2'].fillna('None')
test['BsmtFinType2'] = test['BsmtFinType2'].fillna('None')
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
test['FireplaceQu'] = test['FireplaceQu'].fillna('None')
train['GarageType'] = train['GarageType'].fillna('None')
test['GarageType'] = test['GarageType'].fillna('None')
train['GarageFinish'] = train['GarageFinish'].fillna('None')
test['GarageFinish'] = test['GarageFinish'].fillna('None')
train['GarageQual'] = train['GarageQual'].fillna('None')
test['GarageQual'] = test['GarageQual'].fillna('None')
train['GarageCond'] = train['GarageCond'].fillna('None')
test['GarageCond'] = test['GarageCond'].fillna('None')
train['PoolQC'] = train['PoolQC'].fillna('None')
test['PoolQC'] = test['PoolQC'].fillna('None')
train['Fence'] = train['Fence'].fillna('None')
test['Fence'] = test['Fence'].fillna('None')
train['MiscFeature'] = train['MiscFeature'].fillna('None')
test['MiscFeature'] = test['MiscFeature'].fillna('None')
train["MasVnrType"] = train["MasVnrType"].fillna("None")
test["MasVnrType"] = test["MasVnrType"].fillna("None")

train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
test["MasVnrArea"] = test["MasVnrArea"].fillna(0)

train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)
    
#train = train.drop(train.loc[train['Electrical'].isnull()].index)
#test = test.drop(test.loc[test['Electrical'].isnull()].index)

train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
train = train.drop(['Utilities'], axis=1)
test = test.drop(['Utilities'], axis=1)
train["Functional"] = train["Functional"].fillna("Typ")
test["Functional"] = test["Functional"].fillna("Typ")
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])
train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])

test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    test[col] = test[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    test[col] = test[col].fillna('None')
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])

train['MSSubClass'] = train['MSSubClass'].apply(str)
train['OverallCond'] = train['OverallCond'].astype(str)
train['YrSold'] = train['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)
test['MSSubClass'] = test['MSSubClass'].apply(str)
test['OverallCond'] = test['OverallCond'].astype(str)
test['YrSold'] = test['YrSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)


lbl = LabelEncoder()
scaler = MinMaxScaler()

train_numeric = train._get_numeric_data()
scaler.fit(train_numeric)
train_scaled = pd.DataFrame(scaler.transform(train_numeric),columns = train_numeric.columns.values,index=train.index)


for col in train.columns.values:
   # if col in train_numeric.columns.values :
    #    if col != 'SalePrice':
     #       train[col] = train_scaled[col]
    if train[col].dtype == object :
        lbl.fit(train[col])
        train[col]=lbl.transform(train[col])



test_numeric = test._get_numeric_data()
scaler.fit(test_numeric)
test_scaled = pd.DataFrame(scaler.transform(test_numeric),columns = test_numeric.columns.values,index=test.index)

for col in test.columns.values:    
    #if col in test_numeric.columns.values :
    #    test[col] = test_scaled[col]
    if test[col].dtype == object :
        lbl.fit(test[col])
        test[col]=lbl.transform(test[col])


train_y= np.log(train_y)


numeric_features = train.dtypes[train.dtypes != "object"].index

skewed = train[train_numeric.columns.values].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > abs(0.75)]
skewed = skewed.index

train[skewed] = np.log1p(train[skewed])
test[skewed] = np.log1p(test[skewed])

train = pd.concat([train,train_y],axis=1)


train.to_csv('../../data/train_proc.csv')
test.to_csv('../../data/test_proc.csv')
