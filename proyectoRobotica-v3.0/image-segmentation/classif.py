import numpy as np
from scipy.spatial import distance

class Classifier:
    def __init__(self):
        pass

    def predict(self):
        pass

    def segmenta(self):
        pass


class segEuclid(Classifier):
    def __init__(self, data):
        self.labels = np.array(range(len(data)))
        self.Z = np.zeros(shape=(len(self.labels),np.size(data[0],axis=1)))
        for i in range(len(self.labels)):
            self.Z[i] = np.mean(data[i], axis=0)
        pass

    def predict(self, X):
        if X.shape[2] != self.Z.shape[1]:
            Z = ((self.Z+0.0).T/np.sum(self.Z,axis=1)).T[:,:2]
        else:
            Z = self.Z
        return X.dot(Z.T) - 0.5*np.sum(np.power(Z,2), axis=1)

    def segmenta(self, X):
        return self.labels[np.argmax(self.predict(X), axis=2)]


class segMano(Classifier):
    def __init__(self, data):
        self.labels = np.array(range(len(data)))
        self.Z = np.zeros(shape=(len(self.labels),3))
        self.invCov = np.zeros(shape=(len(self.labels),3,3))
        for i in range(len(self.labels)):
            self.invCov[:,:,i] = np.linalg.inv( np.cov(data[i], rowvar=False, ddof=1) )
            self.Z[i] = np.mean(data[i], axis=0)
        pass

    def predict(self, X):
        R = np.zeros(X.shape)
        for i in range(len(self.labels)):
            R[:,:,i] = np.power(X-self.Z[i],2).dot(self.invCov[i])
        return R
            
    def segmenta(self, X):
        return self.labels[np.argmin(self.predict(X), axis=2)]
