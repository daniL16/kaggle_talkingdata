import pandas as pd
import sys
import numpy as np
import gc
path ='/media/dani/E892136C92133E8E/TFG/data/'
#rows train 184903891
#rows test 18790470
#{'app': 790, 'nip_day_hh': 571, 'channel': 674, 'ip': 532, 'day': 18, 'click_time_timestamp': 371, 'device': 166, 'os': 567}

def processDates(type):
    if type=='test':
        total_rows = 18790470
        step= 18790470
        iters = int(total_rows/step)
        finput=path+'test.csv'
        output = path+'test_proc.csv'
    elif type=='train':
        total_rows = 184903891
        step= 10000000
        iters = int(total_rows/step)+1
        finput=path+'train.csv'
        output = path+'train_proc.csv'
        
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
        #df['year'] =pd.to_numeric(df['click_time'].map(lambda x: x.year),downcast='unsigned')
        #df['month'] =pd.to_numeric(df['click_time'].map(lambda x: x.month),downcast='unsigned')
        #df['next_click'] = (df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - df.click_time).dt.seconds.astype(np.float32)
        #df['next_click'].fillna((df['next_click'].mean()), inplace=True)
        #df['wday']  = pd.to_datetime(df.click_time).dt.dayofweek.astype('uint8')
        print('grouping by ip-day-hour combination....')
        gp = df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
        df = df.merge(gp, on=['ip','day','hour'], how='left')
        del gp; gc.collect()
        print('group by ip-app combination....')
        gp = df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
        df = df.merge(gp, on=['ip','app'], how='left')
        del gp; gc.collect()
        print('group by ip-app-os combination....')
        gp = df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
        df = df.merge(gp, on=['ip','app', 'os'], how='left')
        del gp; gc.collect()
        print("vars and data type....")
        df['qty'] = df['qty'].astype('uint16')
        df['ip_app_count'] = df['ip_app_count'].astype('uint16')
        df['ip_app_os_count'] = df['ip_app_os_count'].astype('uint16')

        #df.drop(['click_id', 'click_time','ip','is_attributed'],1,inplace=True)

        
        df.drop(['click_time'], axis=1, inplace=True)
        df.drop(['day'],axis=1,inplace=True)
        if type=='train':
            df.drop(['attributed_time'],axis=1,inplace=True)
        
        if i == 0:
            df.to_csv(output,index=False)
        else:
            df.to_csv(output, mode='a', header=False,index=False)
    
    
processDates(sys.argv[1])