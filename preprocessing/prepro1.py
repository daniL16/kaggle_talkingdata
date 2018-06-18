import pandas as pd
import sys

path ='/media/dani/E892136C92133E8E/TFG/data/'


def processDates(type):
    if type=='test':
        total_rows = 18790470
        step= 18790470
        iters = int(total_rows/step)
        finput=path+'test.csv'
        output = path+'test_proc1.csv'
    elif type=='train':
        total_rows = 184903891
        step= 10000000
        iters = int(total_rows/step)+1
        finput=path+'train.csv'
        output = path+'train_proc1.csv'
        
    for i in range(0, iters):
        init_row = i*step+1
        print(init_row)
        if i==0:
            df = pd.read_csv(finput,nrows=step)
        else:
            df=pd.read_csv(finput,nrows=step,skiprows=[1,init_row])
        df['click_time'] = pd.to_datetime(df['click_time'])
        df['click_time_timestamp'] = df['click_time'].map(lambda x: x.timestamp())
        df['click_time'] = pd.to_datetime(df['click_time']).dt.date
        df['day']    = pd.to_datetime(df.click_time).dt.day.astype('uint8')
       
        df.drop(['click_time'], axis=1, inplace=True)
        if type=='train':
            df.drop(['attributed_time'],axis=1,inplace=True)
        
        if i == 0:
            df.to_csv(output,index=False)
        else:
            df.to_csv(output, mode='a', header=False,index=False)
    
    
processDates(sys.argv[1])