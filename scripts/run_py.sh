#!/bin/bash

FILE='resultstime_python.txt'

echo -n -e "kNN\t" > $FILE
/usr/bin/time -f "%e  %M" python3 knn.py &>> $FILE
echo 'KNN'
echo -n -e "Linear Regression\t" >> $FILE
/usr/bin/time -f "%e %M" python3 linear_regression.py &>> $FILE
echo 'LR'
echo -n -e "Logistic Regression\t" >> $FILE
/usr/bin/time -f "%e %M" python3 logistic_regression.py &>> $FILE
echo 'LOG'
echo -n -e "Decision Tree\t" >> $FILE
/usr/bin/time -f "%e %;" python3 decision_tree.py &>> $FILE
echo 'DT'
echo -n -e "SVM\t" >> $FILE
/usr/bin/time -f "%e %M" python3 svm.py &>> $FILE
echo 'SVM'
echo -n -e "RandomForest\t" >> $FILE
/usr/bin/time -f "%e %M" python3 random_forest.py &>> $FILE
echo 'RF'
echo -n -e "Gradient Boosting\t" >> $FILE
/usr/bin/time -f "%e %M" python3 gradient_boosting.py &>> $FILE
