#!/bin/bash

FILE='resultstime_python.txt'

echo  -e "Algorithm,Score,Mean Squared,Time,Memory" > $FILE
echo -n -e "kNN\t" &>> $FILE
/usr/bin/time -f "%e  %M" python3 knn.py &>> $FILE

echo -n -e "Decision Tree\t" >> $FILE
/usr/bin/time -f "%e %;" python3 decision_tree.py &>> $FILE
echo -n -e "RandomForest\t" >> $FILE
/usr/bin/time -f "%e %M" python3 random_forest.py &>> $FILE
echo -n -e "Gradient Boosting\t" >> $FILE
/usr/bin/time -f "%e %M" python3 gradient_boosting.py &>> $FILE
echo -n -e "AdaBoost\t" >> $FILE
/usr/bin/time -f "%e %M" python3 adaBoost.py &>> $FILE
echo -n -e "Gaussian\t" >> $FILE
/usr/bin/time -f "%e %M" python3 gaussian.py &>> $FILE
echo -n -e "Neuronal\t" >> $FILE
/usr/bin/time -f "%e %M" python3 neural.py &>> $FILE
echo -n -e "Bagging\t" >> $FILE
/usr/bin/time -f "%e %M" python3 bagging.py &>> $FILE
echo -n -e "Logistic Regression\t" >> $FILE
/usr/bin/time -f "%e %M" python3 logistic_regression.py &>> $FILE
echo -n -e "SVM\t" >> $FILE
/usr/bin/time -f "%e %M" python3 svm.py &>> $FILE
