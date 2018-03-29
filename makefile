path = /media/dani/E892136C92133E8E/TFG/data/predictions

clean:
    rm $(path)/*.csv
submit:
    for submit in $(ls path)
    do
    kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f path/$submit -m "Message"
    done
all: submit clean
