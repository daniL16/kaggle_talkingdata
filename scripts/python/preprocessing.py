import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.preprocessing import OneHotEncoder,Imputer
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../../data/train.csv',header=0,index_col='Id')
test = pd.read_csv('../../data/test.csv',header=0,index_col='Id')

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
train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
test["MasVnrType"] = test["MasVnrType"].fillna("None")
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

train['GarageQual_Ex']=0
train['PoolQC_Fa']=0

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))
    lbl.fit(list(test[c].values)) 
    test[c] = lbl.transform(list(test[c].values))
    
train['GrLivArea'] = np.log(train['GrLivArea'])
test['GrLivArea'] = np.log(test['GrLivArea'])

#train.loc[train['TotalBsmtSF']>0,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
#test.loc[test['TotalBsmtSF']>0,'TotalBsmtSF'] = np.log(test['TotalBsmtSF'])

train['LotArea'] = np.log(train['LotArea'])
test['LotArea'] = np.log(test['LotArea'])

train = pd.get_dummies(train)
test = pd.get_dummies(test)

cols = {'Condition2_RRAe', 'Exterior2nd_Other', 'Condition2_RRAn', 'RoofMatl_Roll', 'Heating_Floor', 'Exterior1st_Stone', 'Heating_OthW', 'HouseStyle_2.5Fin', 'Electrical_Mix', 'RoofMatl_Membran', 'Condition2_RRNn', 'RoofMatl_ClyTile', 'Exterior1st_ImStucc', 'RoofMatl_Metal','GarageQual_Ex','MiscFeature_TenC','PoolQC_Fa'}
for col in cols:
    test[col]=0

train = train.reindex_axis(sorted(train.columns), axis=1)
test = test.reindex_axis(sorted(test.columns), axis=1)

im = Imputer()
names = test.columns.values
test_id = test.index
test = im.fit_transform(test)
test = pd.DataFrame(test,columns=names,index=test_id)


train.to_csv('../../data/train_proc.csv')
test.to_csv('../../data/test_proc.csv')
