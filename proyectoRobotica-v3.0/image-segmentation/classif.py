import numpy as np

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
        self.Z = np.zeros(shape=(len(self.labels),3))
        for i in range(len(self.labels)):
            self.Z[i] = np.mean(data[i], axis=0)
        pass

    def predict(self, X):
        # TODO
        # return X.dot(self.Z.T) - 0.5*np.sum(self.Z**2, axis=1)
        pass

    def segmenta(self, X):
        return self.labels[np.argmax(self.predict(X), axis=1)]

class segMano(Classifier):
    def __init__(self, data):
        self.labels = np.array(range(len(data)))
        self.Z = np.zeros(shape=(len(self.labels),3))
        for i in range(len(self.labels)):
            self.Z[i] = np.mean(data[i], axis=0)
        pass

    def predict(self, X):
        # TODO
        pass

    def predLabel(self, X):
        return self.labels[np.argmax(self.predict(X), axis=1)]
