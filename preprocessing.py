import pandas as pd
path ='/media/dani/E892136C92133E8E/TFG/data/'
def processDates():
    total_rows = 184903891
    step= 10000000
    iters = int(total_rows/step)
    for i in range(0, iters):
        init_row = i*step+1
        df = pd.read_csv(path+'train.csv', skiprows=[1,init_row],nrows=step) 
        df['click_time'] = pd.to_datetime(df['click_time'])
        df['click_time_timestamp'] = df['click_time'].map(lambda x: x.timestamp())
        df['click_time'] = pd.to_datetime(df['click_time']).dt.date
        df['click_time_year'] =pd.to_numeric(df['click_time'].map(lambda x: x.year),downcast='unsigned')
        df['click_time_month'] =pd.to_numeric(df['click_time'].map(lambda x: x.month),downcast='unsigned')
        df['click_time_day'] =pd.to_numeric(df['click_time'].map(lambda x: x.day),downcast='unsigned')
        df.drop(['click_time'], axis=1, inplace=True)
        if i == 0:
            df.to_csv(path+'procDatestrain.csv',index=False)
        else:
            df.to_csv(path+'procDatestrain.csv', mode='a', header=False,index=False)
    
 
 
 
   
processDates()