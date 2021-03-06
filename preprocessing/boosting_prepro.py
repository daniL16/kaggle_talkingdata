import pandas as pd
import xgboost as xgb
import sys
import gc
from sklearn.cross_validation import train_test_split

if sys.argv[1] == '0' :
    train = '/media/dani/E892136C92133E8E/TFG/data/train.csv'
    test_file = '/media/dani/E892136C92133E8E/TFG/data/test.csv'
else:
    train = '/media/dani/E892136C92133E8E/TFG/data/train_proc'+sys.argv[1]+'.csv'
    test_file = '/media/dani/E892136C92133E8E/TFG/data/test_proc'+sys.argv[1]+'.csv'
x = pd.read_csv(train,nrows=25000000)
y = x['is_attributed']
x.drop(['is_attributed'], axis=1, inplace=True)
params={'eval_metric': 'auc','silent': True}


x1, x2, y1, y2 = train_test_split(x, y, test_size=0.1, random_state=99)
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params,xgb.DMatrix(x1, y1), 100, watchlist, maximize=True, verbose_eval=10)
del x,y,x1,x2,y1,y2,watchlist
gc.collect()

print(model.get_fscore())
print("making predictions")
test = pd.read_csv(test_file)
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop(['click_id'], axis=1, inplace=True)


sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('/media/dani/E892136C92133E8E/TFG/predictions/proc'+sys.argv[1]+'.csv',index=False)
