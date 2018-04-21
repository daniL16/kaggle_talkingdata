import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
path ='/media/dani/E892136C92133E8E/TFG/data/'

def imbalacedClass():
    train = pd.read_csv(path+'test.csv',usecols=['is_attributed'])
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
    train = pd.read_csv(path+'test.csv',usecols=[field])
    sns.distplot(train[field])
    plt.savefig('normalDist'+field+'test.png')
    #plt.show()

def missingData(field):
    train = pd.read_csv(path+'test.csv',usecols=[field])
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20))
    
def dataDistribution(field):
    train = pd.read_csv(path+'test.csv',usecols=[field])
    print(train.head())
    plt.hist(train.values)
    plt.savefig(field+'_distribution.png')
    #plt.show()
    

features = ['click_time_day','click_time_month','click_time_year']
#for feat in features:
#    dataDistribution(feat)
#    missingData(feat)
#normalDistribution('click_time_timestamp')
imbalacedClass()
