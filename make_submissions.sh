#!/bin/bash
pwd = ${pwd}
cd   /media/dani/E892136C92133E8E/TFG/data/predictions
for submit in $(ls )
do
kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f $submit -m "Message"
done
cd $(pwd)