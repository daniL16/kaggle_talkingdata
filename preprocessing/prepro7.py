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
        output = path+'test_proc7.csv'
    elif type=='train':
        total_rows = 184903891
        step= 10000000
        iters = int(total_rows/step)+1
        finput=path+'train.csv'
        output = path+'train_proc7.csv'
        
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
        
        print('group by ip-app combination....')
        gp = df[['ip','app','channel']].groupby(by=['ip', 'app'])[['channel']].mean().reset_index().rename(index=str, columns={'channel': 'ip_app_media'})
        df = df.merge(gp, on=['ip','app'], how='left')
        del gp; gc.collect()
        
        print('group by ip-app-os combination....')
        gp = df[['ip','app','os','channel']].groupby(by=['ip', 'app','os'])[['channel']].mean().reset_index().rename(index=str, columns={'channel': 'ip_app_os_media'})
        df = df.merge(gp, on=['ip','app','os'], how='left')
        del gp; gc.collect()
        
        df['ip_app_media'] = df['ip_app_media'].astype('uint16')
        df['ip_app_os_media'] = df['ip_app_os_media'].astype('uint16')
       
        df.drop(['click_time'], axis=1, inplace=True)
        
        if type=='train':
            df.drop(['attributed_time'],axis=1,inplace=True)
        
        if i == 0:
            df.to_csv(output,index=False)
        else:
            df.to_csv(output, mode='a', header=False,index=False)
    
    
processDates(sys.argv[1])
