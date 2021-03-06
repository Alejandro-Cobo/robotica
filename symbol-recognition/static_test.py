import numpy as np
import sklearn.neighbors
import sklearn.cross_validation as cv

from lib import hu_moments as hu
from lib import mahalanobis

data, labels = hu.get_db()
loo = cv.LeaveOneOut(len(labels))

# 1-NN
aciertos = 0
for tr, te in loo:
    knn = sklearn.neighbors.KNeighborsClassifier(1,algorithm='brute',metric='euclidean')
    knn = knn.fit(data[tr],labels[tr])
    ypred = knn.predict(data[te])
    aciertos += (ypred != labels[te])
print("k-NN:")
print("\t-- Porcentaje de acierto: " + str((1-(float(aciertos)/len(labels)))*100) + "%")

# Mahalanobis
aciertos = 0
for tr, te in loo:
    maha = mahalanobis.classifMahalanobis()
    maha = maha.fit(data[tr],labels[tr])
    ypred = maha.predict(data[te])
    aciertos += (ypred != labels[te])
print("Mahalanobis:")
print("\t-- Porcentaje de acierto: " + str((1-(float(aciertos)/len(labels)))*100) + "%")
