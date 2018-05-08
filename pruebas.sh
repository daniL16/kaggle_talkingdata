#!/bin/bash


#python3 preprocessing.py test
#python3 preprocessing.py train


#python3 boosting.py 30000000 ada boosting30f.csv | tee log/bostexact30.log
#python3 boosting.py 40000000 ada boosting40f.csv | tee log/bostexact40.log
#python3 boosting.py 35000000 ada boosting35f.csv | tee log/bostexact35.log

#echo 'boost'

python3 rusboost.py 130000000 rus130.csv ada | tee log/rus130rus.log
echo "rus"

python3 cusboost.py 5000000 cus5.csv  | tee log/cus5.log
python3 cusboost.py 3000000 cus3.csv  | tee log/cus3.log
python3 cusboost.py 1000000 cus1.csv  | tee log/cus1.log
python3 cusboost.py 10000000 cus5.csv  | tee log/cus10.log
python3 cusboost.py 15000000 cus5.csv  | tee log/cus15.log
echo "cus"
#python3 boosting.py 100000000 knn knn100.csv  &> log/100knn.log
#python3 boosting.py 100000000 centroids centroids100.csv  &> log/100cen.log



#python3 boosting.py 150000000 knn knn150.csv  &> log/150knn.log
#python3 boosting.py 150000000 knn knn150.csv  &> log/150knn.log
#python3 boosting.py 150000000 centroids centroids150.csv  &> log/150cen.log

#python3 boosting.py 175000000 random rand175pri.csv  &> log/175rand.log
#python3 boosting.py 175000000 knn knn175.csv  &> log/175knn.log
#python3 boosting.py 175000000 centroids centroids175.csv  &> log/175cen.log



