import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

path ='/media/dani/E892136C92133E8E/TFG/data/'
features = ['app','ip','os','device','channel','click_time','attributed_time']

def imbalacedClass():
    train = pd.read_csv(path+'train.csv',usecols=['is_attributed'])
    negatives = train[train['is_attributed']==0]
    positives = train[train['is_attributed']==1]
    impr = ["false","true"]
    vol = [negatives.count(),positives.count()]
    expl =(0, 0.05)
    plt.pie(vol, explode=expl, labels=impr, autopct='%1.1f%%', shadow=True)
    plt.title("Balanceo de Clases", bbox={"facecolor":"0.8", "pad":5})
    plt.legend()
    plt.savefig('imbalacing_test.png')

def normalDistribution(field):
    train = pd.read_csv(path+'train.csv',usecols=[field])
    sns.distplot(train[field])
    plt.savefig('normalDist'+field+'test.png')

def missingData(field):
    train = pd.read_csv(path+'train.csv',usecols=[field])
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20))
        
def dataDistribution(field):
    train = pd.read_csv(path+'train.csv',usecols=[field])
    plt.hist(train.values)
    plt.title("Distribución de "+ field)
    plt.legend()
    plt.savefig(field+'_distribution.png')
def correlation():
    df_train = pd.read_csv(path+'train.csv',usecols=['app','ip','os','device','channel'],nrows=50000000)
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df_train.corr()
    sns.set(font_scale=1.5)
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
    ax.tick_params(labelsize=20)
    plt.savefig('../TFGdoc/img/correlation.png')
    plt.show()
    
    
for feat in features:
    dataDistribution(feat) 
    missingData(feat)
normalDistribution('click_time_timestamp')
imbalacedClass()
correlation()
