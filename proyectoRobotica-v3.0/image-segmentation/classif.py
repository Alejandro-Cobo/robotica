# coding=UTF-8
from sklearn.qda import QDA

class segQDA():
    def __init__(self, data, labels):
        self.gnb = QDA()
        self.gnb.fit(data, labels)

    def segmenta(self,X):
        return self.gnb.predict(X)
