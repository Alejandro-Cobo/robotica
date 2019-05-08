# coding=UTF-8
from sklearn import discriminant_analysis as da

class segQDA():
    def __init__(self, data, labels):
        self.qda = da.QuadraticDiscriminantAnalysis()
        self.qda.fit(data, labels)

    def segmenta(self,X):
        return self.qda.predict(X)