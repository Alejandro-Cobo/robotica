import numpy as np
import sklearn.neighbors

class segEuclid():
    def __init__(self, data, labels):
        # 0: rojo
        # 1: verde
        # 2: azul
        labels_unique = np.unique(labels)
        self.Z = np.zeros((labels_unique.shape[0], data.shape[1]))
        for i in labels_unique:
            self.Z[i] = np.mean(data[labels==i], 0)

    def segmenta(self,X):
        return np.argmax(X.dot(self.Z.T)-0.5*np.sum(np.power(self.Z, 2), 1), 1)

class segKNN():
    def __init__(self, data, labels):
        # 0: rojo
        # 1: verde
        # 2: azul
        self.neigh = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
        self.neigh.fit(data, labels)

    def segmenta(self,X):
        return self.neigh.predict(X)