from sklearn import tree
import numpy as np
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

   