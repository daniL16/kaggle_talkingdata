import pandas as pd

from sklearn.neighbors import LocalOutlierFactor

data = pd.read_csv('../../data/train_proc.csv',header=0)

rows = data.shape[0]
clf=LocalOutlierFactor(contamination=0.01)
y_pred = clf.fit_predict(data)  
y_pred=pd.DataFrame(y_pred,columns=['Outlier'])
pred = pd.concat([data.iloc[:,0],y_pred], axis=1)
data = data.drop(pred[(pred['Outlier']==-1)].index)

rows = rows-data.shape[0]
print("Rows deleted: %s"%rows)

data.to_csv('../../data/train_outliers.csv',index=False)


