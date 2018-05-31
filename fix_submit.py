#script para ajustar el archivo de prediciones
#para ajustar las predicciones a [0,1] 
import pandas as pd
import sys

path='/media/dani/E892136C92133E8E/TFG/predictions/'
data = pd.read_csv(path+sys.argv[1])
atr = data['is_attributed']
atr[atr<0] = 0
atr[atr>1] = 1
data['is_attributed'] = atr
data.to_csv(path+sys.argv[1],index=False)