from sklearn import svm
from sklearn import tree
from math import log
import random
import numpy as np
import pandas as pd
import imblearn
from imblearn.under_sampling import RandomUnderSampler
class RUSBoost:


    def __init__(self, instances, labels, base_classifier, n_classifier, balance):
        
        self.w_update=[]
        self.clf = []
        self.n_classifier = n_classifier
        for i in range(n_classifier):
            self.clf.append(base_classifier)
        self.rate = balance
        self.X = instances
        self.Y = labels
        
        # initialize weight
        self.weight = []
        self.init_w = 1.0/len(self.X)
        for i in range(len(self.X)):
            self.weight.append(self.init_w)
    
    def classify(self, instance):
        
        positive_score = 0 # in case of +1
        negative_score = 0 # in case of 0
        prediction=[]
        for i in range(len(instance)):
            current_instance = np.array(instance[i]).reshape(1,-1)
            for k in range(self.n_classifier):
                if self.clf[k].predict(current_instance) == 1:
                    positive_score += log(1/self.w_update[k])
                else:
                    negative_score += log(1/self.w_update[k])
            if negative_score <= positive_score:
                prediction.append(1)
            else:
                prediction.append(0)

        return prediction
            
        
    def learning(self):
        
        k = 0
        while k < self.n_classifier:
            print (k)
            sampled = self.undersampling()
            sampled_X = []
            sampled_Y = []
            sampled_weight = []
            
            for s in sampled:
                sampled_X.append(s[1])
                sampled_Y.append(s[2])
                sampled_weight.append(self.weight[s[0]])
                
            self.clf[k].fit(sampled_X, sampled_Y, sampled_weight)
           
   
            loss = 0
            print("calculating loss")
            for i in range(len(self.X)):
                if self.Y[i] == self.clf[k].predict(np.array(self.X[i]).reshape(1,-1)):
                    continue
                else:
                    loss += self.weight[i]
    
            self.w_update.append(loss/(1-loss))
        
            for i in range(len(self.weight)):
                if loss == 0:
                    self.weight[i] = self.weight[i]
                elif self.Y[i] == self.clf[k].predict(np.array(self.X[i]).reshape(1,-1)):
                    self.weight[i] = self.weight[i] * (loss / (1 - loss))
                       
            sum_weight = 0
            for i in range(len(self.weight)):
                sum_weight += self.weight[i]
              
            for i in range(len(self.weight)):
                self.weight[i] = self.weight[i] / sum_weight
            k = k + 1
     
            
    def undersampling(self):
       
        print('undersampling')
        '''Check the major class'''
     
        true_list = []
        for i in range(len(self.Y)):
            if self.Y[i]:
                true_list.append([i,self.X[i],1])
        i=0
        tope = 0.65*len(self.X)-len(true_list)
        new_list = []
        while i < tope:
            k = int(tope * np.random.random_sample())
            if self.Y[k] == 0:
                new_data = [k,self.X[k],0]
                new_list.append(new_data)
                i += 1
        print (len(new_list))
        print('end undersampling')
        return new_list+true_list