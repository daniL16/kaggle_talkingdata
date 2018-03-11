#!/bin/bash

for submit in $(ls ./predictions)
do
kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f ./predictions/$submit -m "Message"
done
