import numpy as np

class segEuclid():
    def __init__(self, data):
        self.Z = np.zeros((3,2))
        means = np.zeros((3,3))
        for i in range(len(data)):
            means[i] = np.mean(data[i],0)
        self.Z = ((means).T/np.sum(means,1)).T[:,:2]

    def segmenta(self, X):
        return np.argmax(X.dot(self.Z.T)-0.5*np.sum(np.power(self.Z, 2), 1), 2)