#!/bin/bash
python3 boosting.py 150000000 random rand150.csv
python3 boosting.py 150000000 knn knn150.csv
python3 boosting.py 150000000 centroids centroids150.csv

python3 boosting.py 175000000 random rand175pri.csv
python3 boosting.py 175000000 knn knn175.csv
python3 boosting.py 175000000 centroids centroids175.csv
