import numpy as np
import sklearn.neighbors
import sklearn.cross_validation as cv

import db_hu_moments as hu
import mahalanobis

data, labels = hu.get_hu_moments()

# 1-NN
loo = cv.LeaveOneOut(len(labels))
errors = []
for tr, te in loo:
    knn = sklearn.neighbors.KNeighborsClassifier(1)
    knn = knn.fit(data[tr],labels[tr])
    ypred = knn.predict(data[te])
    errors.append((sum(ypred==labels[te])+0.0)/len(labels[te]))
print("k-NN:")
print("\t- Porcentaje de acierto: " + str(sum(errors)/len(errors)*100) + "%")

# Mahalanobis
loo = cv.LeaveOneOut(len(labels))
errors = []
for tr, te in loo:
    maha = mahalanobis.classifMahalanobis()
    maha = maha.fit(data[tr],labels[tr])
    ypred = maha.predict(data[te])
    errors.append((sum(ypred==labels[te])+0.0)/len(labels[te]))
print("Mahalanobis:")
print("\t- Porcentaje de acierto: " + str(sum(errors)/len(errors)*100) + "%")
