import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../../data/train.csv',header=0)
test = pd.read_csv('../../data/test.csv',header=0)
sns.distplot(train['SalePrice']);
"""
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();
"""

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
test = test.drop((missing_data[missing_data['Total'] > 1]).index,1)
test = test.drop(test.loc[test['Electrical'].isnull()].index)

#train['SalePrice'] = np.log(train['SalePrice'])
train['GrLivArea'] = np.log(train['GrLivArea'])
test['GrLivArea'] = np.log(test['GrLivArea'])


train = pd.get_dummies(train)
test = pd.get_dummies(test)

cols = {'Condition2_RRAe', 'Exterior2nd_Other', 'Condition2_RRAn', 'Utilities_NoSeWa', 'RoofMatl_Roll', 'Heating_Floor', 'Exterior1st_Stone', 'Heating_OthW', 'HouseStyle_2.5Fin', 'Electrical_Mix', 'RoofMatl_Membran', 'Condition2_RRNn', 'RoofMatl_ClyTile', 'Exterior1st_ImStucc', 'RoofMatl_Metal'}
for col in cols:
    test[col]=0

train = train.reindex_axis(sorted(train.columns), axis=1)
test = test.reindex_axis(sorted(test.columns), axis=1)

train.to_csv('../../data/train_proc.csv',index=False)
test.to_csv('../../data/test_proc.csv',index=False)
