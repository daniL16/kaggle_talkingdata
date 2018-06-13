from sklearn import svm
from sklearn import tree
from math import log
import random
import numpy as np
import pandas as pd
import imblearn
import sys
from imblearn.under_sampling import RandomUnderSampler


class RUSBoost:
    def __init__(self, M, depth, verbose=False):
        self.M = M
        self.depth = depth
        self.undersampler = RandomUnderSampler(return_indices=True,replacement=False)
        self.verbose = verbose

    def fit(self, X, Y):
        self.models = []
        self.alphas = []

        N, _ = X.shape
        W = np.ones(N) / N

        for m in range(self.M):
            if self.verbose:
                print ("Iteracion " + str(m))
                
            tr = tree.DecisionTreeClassifier(max_depth=self.depth, splitter='best')

            X_undersampled, y_undersampled, chosen_indices = self.undersampler.fit_sample(X, Y)

            tr.fit(X_undersampled, y_undersampled, sample_weight=W[chosen_indices])

            P = tr.predict(X)

            err = np.sum(W[P != Y])

            if err > 0.5:
                m = m - 1
            if err <= 0:
                err = 0.0000001
            else:
                try:
                    if (np.log(1 - err) - np.log(err)) == 0 :
                        alpha = 0
                    else:
                        alpha = 0.5 * (np.log(1 - err) - np.log(err))
                    W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1
                except:
                    alpha = 0
                    # W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1

                self.models.append(tr)
                self.alphas.append(alpha)
                W = W.values
    def predict(self, X):
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tr in zip(self.alphas, self.models):
            FX += alpha * tr.predict(X)
      
        return np.sign(FX)

    def predict_proba(self, X):
        proba = sum(tr.predict_proba(X) * alpha for tr , alpha in zip(self.models,self.alphas) )


        proba = np.array(proba)


        proba = proba / sum(self.alphas)

        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba = proba /  normalizer
        print(proba)
        return proba[:,0]