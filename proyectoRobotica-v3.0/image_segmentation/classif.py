# coding=UTF-8
import numpy as np
from sklearn import discriminant_analysis as da

class segQDA():
    def __init__(self, data, labels):
        self.gnb = da.QuadraticDiscriminantAnalysis()
        self.gnb.fit(data, labels)

    def segmenta(self,X):
        return self.gnb.predict(X)