import numpy as np
import sklearn.neighbors
import sklearn.cross_validation as cv

import hu_moments as hu
import mahalanobis

import cv2
import binarize_image as bin

data, labels = hu.get_db_hu()
loo = cv.LeaveOneOut(len(labels))

# 1-NN
errors = 0.0
for tr, te in loo:
    knn = sklearn.neighbors.KNeighborsClassifier(1)
    knn = knn.fit(data[tr],labels[tr])
    ypred = knn.predict(data[te])
    errors += ypred != labels[te]
print("k-NN:")
print("\t-- Porcentaje de acierto: " + str((1-(errors/len(labels)))*100) + "%")

# Mahalanobis
errors = 0.0
for tr, te in loo:
    maha = mahalanobis.classifMahalanobis()
    maha = maha.fit(data[tr],labels[tr])
    ypred = maha.predict(data[te])
    errors += ypred != labels[te]
print("Mahalanobis:")
print("\t-- Porcentaje de acierto: " + str((1-(errors/len(labels)))*100) + "%")
