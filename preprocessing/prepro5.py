import pandas as pd
import sys
import numpy as np
import gc
path ='/media/dani/E892136C92133E8E/TFG/data/'


def processDates(type):
    if type=='test':
        total_rows = 18790470
        step= 18790470
        iters = int(total_rows/step)
        finput=path+'test.csv'
        output = path+'test_proc5.csv'
    elif type=='train':
        total_rows = 184903891
        step= 10000000
        iters = int(total_rows/step)+1
        finput=path+'train.csv'
        output = path+'train_proc5.csv'
        
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
        df['hour']    = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
        df['day']    = pd.to_datetime(df.click_time).dt.day.astype('uint8')
      
        print('grouping by ip-day-hour combination....')
        gp = df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_day_hour_count'})
        df = df.merge(gp, on=['ip','day','hour'], how='left')
        del gp; gc.collect()
        
        print('grouping by ip-day combination....')
        gp = df[['ip','day','channel']].groupby(by=['ip','day'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_day_count'})
        df = df.merge(gp, on=['ip','day'], how='left')
        del gp; gc.collect()
        
         
        df['ip_day_hour_count'] = df['ip_day_hour_count'].astype('uint16')
        df['ip_day_count'] = df['ip_day_count'].astype('uint16')
        
        df.drop(['click_time'], axis=1, inplace=True)
        df.drop(['day'],axis=1,inplace=True)
        if type=='train':
            df.drop(['attributed_time'],axis=1,inplace=True)
        
        if i == 0:
            df.to_csv(output,index=False)
        else:
            df.to_csv(output, mode='a', header=False,index=False)
    
    
processDates(sys.argv[1])