import pandas as pd
from sklearn import preprocessing

data = pd.read_csv('../data/test.csv',header=0)
#FILL NA
data = data.fillna(method='ffill')
data = data.fillna(method='bfill')

data=data.iloc[:,1:]
print(data.head())
le = preprocessing.LabelEncoder()
for col in data.columns:
    if(data[col].dtype=='object'):
        data[col]=le.fit_transform(data[col])
data.to_csv('../data/test_proc.csv')


