# coding=UTF-8
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
import numpy as np
import scipy.linalg as la
import sklearn.neighbors
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.simplefilter("ignore",DeprecationWarning)
from sklearn.qda import QDA

class segEuclid():
    def __init__(self, data=None, labels=None, save=False, file=None):
        if file != None:
            self.Z = np.load(file)
            return
        labels_unique = np.unique(labels)
        self.Z = np.zeros((labels_unique.shape[0], data.shape[1]))
        for i in labels_unique:
             self.Z[i] = np.mean(data[labels==i], 0)
        if save:
            np.save("segEuclid_config", self.Z)

    def segmenta(self,X):
        return np.argmax(X.dot(self.Z.T)-0.5*np.sum(np.power(self.Z, 2), 1), 1)

class segKNN():
    def __init__(self, data, labels):
        # 0: rojo
        # 1: verde
        # 2: azul
        self.neigh = sklearn.neighbors.KNeighborsClassifier(n_jobs=2)
        self.neigh.fit(data, labels)

    def segmenta(self,X):
        return self.neigh.predict(X)

class segGNB():
    def __init__(self, data, labels):
        # 0: rojo
        # 1: verde
        # 2: azul
        self.gnb = GaussianNB()
        self.gnb.fit(data, labels)

    def segmenta(self,X):
        return self.gnb.predict(X)

class segQDA():
    def __init__(self, data, labels):
        # 0: rojo
        # 1: verde
        # 2: azul
        self.gnb = QDA()
        self.gnb.fit(data, labels)

    def segmenta(self,X):
        return self.gnb.predict(X)
