#!/bin/bash

#python3 ./preprocessing/prepro1.py train; 
#python3 ./preprocessing/prepro1.py test;
#python3 ./preprocessing/prepro2.py train;
#python3 ./preprocessing/prepro2.py test
#python3 ./preprocessing/prepro3.py train; 
#python3 ./preprocessing/prepro3.py test
#python3 ./preprocessing/prepro4.py train; 
#python3 ./preprocessing/prepro4.py test
#python3 ./preprocessing/prepro5.py train; 
#python3 ./preprocessing/prepro5.py test
python3 ./preprocessing/boosting_prepro.py 2 | tee preprocessing/proc2.log
python3 ./preprocessing/boosting_prepro.py 1 | tee preprocessing/proc1.log
python3 ./preprocessing/boosting_prepro.py 3 | tee preprocessing/proc3.log
python3 ./preprocessing/boosting_prepro.py 4 | tee preprocessing/proc4.log
python3 ./preprocessing/boosting_prepro.py 5 | tee preprocessing/proc5.log






