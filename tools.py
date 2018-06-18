import pandas as pd


class Tools:
    def __init__(self):
        self.path = '/media/dani/E892136C92133E8E/TFG/data/'
        self.features = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed', 'click_time_timestamp', 'hour', 'day',
            'ip_app_media', 'ip_app_os_media', 'ip_app_count', 'ip_day_hour_count']
        self.int_features = ['ip', 'app', 'device', 'os', 'channel', 'click_time_timestamp', 'hour', 'day', 'ip_app_media',
                'ip_app_os_media', 'ip_app_count', 'ip_day_hour_count']
        self.bool_features = ['is_attributed']
        
    # funcion auxiliar para monitorizar el uso de memoria
    def mem_usage(self,obj):
        if isinstance(obj, pd.DataFrame):
            usage_b = obj.memory_usage(deep=True).sum()
        else:
            usage_b = obj.memory_usage(deep=True)
        usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
        return "{:03.2f} MB".format(usage_mb)
    


    # funciones para optimizar la lectura de los conjuntos de train/test
    def low_ram_train_read(self,nrows, init_row=0):
        for feature in self.features:
            train_chunk = pd.read_csv(self.path + 'train_proc.csv', usecols=[feature], nrows=nrows)
            if feature in self.int_features:
                train_chunk = pd.to_numeric(train_chunk[feature], downcast='unsigned')
            elif feature in self.bool_features:
                train_chunk = train_chunk[feature].astype('bool')
            if feature == 'ip':
                train = pd.DataFrame(train_chunk)
            else:
                train[feature] = train_chunk
        return train


    def low_ram_test_read(self):
        feats = self.features
        feats.insert(0, 'click_id')
        feats.remove('is_attributed')
        for feature in feats:
            test_chunk = pd.read_csv(self.path + 'test_proc.csv', usecols=[feature])
            if feature in self.int_features:
                test_chunk = pd.to_numeric(test_chunk[feature], downcast='unsigned')
            if feature == 'click_id':
                test = pd.DataFrame(test_chunk)
            else:
                test[feature] = test_chunk
        return test
