#!/bin/bash

FILE='../results_R.txt'

echo  -e "Algorithm,Time,Memory" > $FILE
echo -n -e "kNN\t" &>> $FILE
/usr/bin/time -f "%e  %M" Rscript knn.R &>> $FILE
echo -n -e "Decision Tree\t" >> $FILE
/usr/bin/time -f "%e %;" Rscript decisiontree.R &>> $FILE
echo -n -e "Neural\t" >> $FILE
/usr/bin/time -f "%e %M" Rscript neural.R &>> $FILE
echo -n -e "SVR\t" >> $FILE
/usr/bin/time -f "%e %M" Rscript svr.R &>> $FILE
echo -n -e "RandomForest\t" >> $FILE
/usr/bin/time -f "%e %M" Rscript rf.R &>> $FILE
echo -n -e "Gradient Boosting\t" >> $FILE
/usr/bin/time -f "%e %M" Rscript boosting.R &>> $FILE
echo -n -e "Bagging\t" >> $FILE
/usr/bin/time -f "%e %M" Rscript bagging.R &>> $FILE
echo -n -e "Logistic Regression\t" >> $FILE
/usr/bin/time -f "%e %M" Rscript logistic.R &>> $FILE
echo -n -e "Gaussian\t" >> $FILE
/usr/bin/time -f "%e %M" Rscript gaussian.R &>> $FILE


