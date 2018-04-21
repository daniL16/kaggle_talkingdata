import pandas as pd
import sys

path ='/media/dani/E892136C92133E8E/TFG/data/'
#rows train 184903891
#rows test 18790470
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
        df = pd.read_csv(finput,nrows=step) 
        df['click_time'] = pd.to_datetime(df['click_time'])
        df['click_time_timestamp'] = df['click_time'].map(lambda x: x.timestamp())
        df['click_time'] = pd.to_datetime(df['click_time']).dt.date
        df['hour']    = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
        df['day']    = pd.to_datetime(df.click_time).dt.day.astype('uint8')
        df['year'] =pd.to_numeric(df['click_time'].map(lambda x: x.year),downcast='unsigned')
        df['month'] =pd.to_numeric(df['click_time'].map(lambda x: x.month),downcast='unsigned')
        
        df.drop(['click_time'], axis=1, inplace=True)
        df.drop(['attributed_time'],axis=1,inplace=True)
        
        if i == 0:
            df.to_csv(path+'procDatest.csv',index=False)
        else:
            df.to_csv(path+'procDatest.csv', mode='a', header=False,index=False)
    
    
processDates(sys.argv[1])