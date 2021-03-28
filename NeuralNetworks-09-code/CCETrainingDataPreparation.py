"""

Author: Sayed Kamaledin Ghiasi-Shirazi
Year:   2019

"""

import numpy as np

class TrainingData:
    def __init__(self, X, y):
        self.C  = len(np.unique(y))
        self.X = X.copy()
        self.y = y.squeeze()

    def findSubclasses(self, K, clusAlg=None):
        self.N, self.d = self.X.shape
        self.K = K
        self.subclassMeans = np.zeros ([self.C*self.K, self.d])

        for c in range (self.C):
            Xi = self.X[self.y == c,:]
            clusAlg.n_clusters = self.K
            clusAlg.fit(Xi)
            skip = 0
            for k in range(self.K):
                self.subclassMeans[c*self.K+k,:] = np.mean(Xi[clusAlg.labels_ == k, :],axis = 0)

        self.L = self.C * self.K
        self.label = np.zeros(self.L)
        idx = 0
        for c in range (self.C):
            for k in range (self.K):
                self.label[idx] = c
                idx += 1
        
    def setSubclasses(self, subClasses):
        self.N, self.d = self.X.shape
        self.K = subClasses.shape[0] // self.C
        self.L = self.C * self.K
        self.label = np.zeros(self.L)
        self.subclassMeans = subClasses.copy()
        idx = 0
        for c in range (self.C):
            for k in range (self.K):
                self.label[idx] = c
                idx += 1
