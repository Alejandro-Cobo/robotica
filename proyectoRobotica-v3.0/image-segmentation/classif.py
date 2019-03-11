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
        return X.dot(Z.T) - 0.5*np.sum(Z**2, axis=1)

    def segmenta(self, X):
        return self.labels[np.argmax(self.predict(X), axis=-1)]

class segMano(Classifier):
    def __init__(self, data):
        self.labels = np.array(range(len(data)))
        self.Z = np.zeros(shape=(len(self.labels),3))
        for i in range(len(self.labels)):
            self.Z[i] = np.mean(data[i], axis=0)
        pass

    def predict(self, X):
        res = np.zeros(shape=X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    u = X[i,j,:]
                    v = self.Z[k]
                    VI = np.linalg.pinv(np.cov(np.array([u,v]).T))
                    res[i,j,k] = distance.mahalanobis(u,v,VI)
        return res
            
    def segmenta(self, X):
        return self.labels[np.argmax(self.predict(X), axis=-1)]
