#script para automatizar las subidas de predicciones a kaggle

#!/bin/bash
pwd = ${pwd}
cd   /media/dani/E892136C92133E8E/TFG/predictions
for submit in $(ls )
do
kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f $submit -m "Message"
#rm $submit
done
cd $(pwd)
