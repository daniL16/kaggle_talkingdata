#forked from https://github.com/farshidrayhanuiu/CUSBoost

from sklearn import tree
import numpy as np
from imblearn.under_sampling import ClusterCentroids


class CUSBoost:
    def __init__(self,instances,labels,k):
        self.weight = []
        self.X= instances
        self.Y = labels
        self.k = k
        self.init_w = 1.0/len(self.X)
        for i in range(len(self.X)):
            self.weight.append(self.init_w)
    def learning(self):
        self.models = []
        self.alphas = []

        N, _ = self.X.shape
        W = np.ones(N) / N
        for i in range(self.k):
            print(i)
            cus = ClusterCentroids(ratio='majority')
            x_undersampled,y_undersampled= cus.fit_sample(self.X,self.Y)
            cl = tree.DecisionTreeClassifier( splitter='best')
            cl.fit(x_undersampled, y_undersampled)

            P = cl.predict(self.X)
            
            err = np.sum(W[P != self.Y])

            if err > 0.5:
                i = i - 1
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

                self.models.append(cl)
                self.alphas.append(alpha)
                

    def predict(self,X):
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tr in zip(self.alphas, self.models):
            FX += alpha * tr.predict(X)
      
        return np.sign(FX)
