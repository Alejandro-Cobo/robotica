import numpy as np
import sklearn.neighbors
import sklearn.cross_validation as cv

import db_hu_moments

def evalClassifier(classif, xtrain, ytrain, xtest, ytest):
    ypred = classif.fit(xtrain,ytrain).predict(xtest)
    return sum(ypred==ytest) / len(ytest)

data, labels = db_hu_moments.get_hu_moments()

# 1-NN
loo = cv.LeaveOneOut(len(labels))
errors = []
for tr, te in loo:
    knn = sklearn.neighbors.KNeighborsClassifier(1)
    knn = knn.fit(data[tr],labels[tr])
    ypred = knn.predict(data[te])
    errors.append((sum(ypred==labels[te])+0.0)/len(labels[te]))
print("-- k-NN:")
print("\t-- " + str((sum(errors)+0.0)/len(errors)))

# Mahalanobis
# TODO
