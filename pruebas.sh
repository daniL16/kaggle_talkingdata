#!/bin/bash

python3 ./preprocessing/prepro1.py train; python3 ./preprocessing/prepro1.py test
python3 ./preprocessing/prepro2.py train; python3 ./preprocessing/prepro2.py test
python3 ./preprocessing/prepro3.py train; python3 ./preprocessing/prepro3.py test
python3 ./preprocessing/prepro4.py train; python3 ./preprocessing/prepro4.py test
python3 ./preprocessing/prepro5.py train; python3 ./preprocessing/prepro5.py test
python3 ./preprocessing/boosting_prepro.py train_proc2.csv | tee preprocessing/proc2.log
python3 ./preprocessing/boosting_prepro.py train_proc1.csv | tee preprocessing/proc1.log
python3 ./preprocessing/boosting_prepro.py train_proc3.csv | tee preprocessing/proc3.log
python3 ./preprocessing/boosting_prepro.py train_proc4.csv | tee preprocessing/proc4.log
python3 ./preprocessing/boosting_prepro.py train_proc5.csv | tee preprocessing/proc5.log






