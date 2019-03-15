# coding=UTF-8
import cv, cv2
from scipy.misc import imread, imsave
import numpy as np
from sklearn import cross_validation as cv
import time

import classif as cl

# Se crean los datasets de entrenamiento y test
imNp = imread('linea.png')
markImg = imread('lineaMarcada.png')
data_marca=imNp[np.where(np.all(np.equal(markImg,(255,0,0)),2))]
data_fondo=imNp[np.where(np.all(np.equal(markImg,(0,255,0)),2))]
data_linea=imNp[np.where(np.all(np.equal(markImg,(0,0,255)),2))]

labels_marca = np.zeros(data_marca.shape[0],np.int8) + 2
labels_fondo = np.zeros(data_fondo.shape[0],np.int8)
labels_linea = np.ones(data_linea.shape[0],np.int8)

data = np.concatenate([data_marca, data_fondo, data_linea])
data = ((data+0.0) / np.sum(data,1)[:,np.newaxis])[:,:2]
labels = np.concatenate([labels_marca,labels_fondo, labels_linea])

skf = cv.StratifiedKFold(labels, n_folds=5)
for tr,te in skf:
    break

############# CLASIFICADOR EUCLÍDEO #############
# ENTRENAMIENTO
start = time.time()
seg = cl.segEuclid(data[tr], labels[tr])
end = time.time()
print("- Clasificador euclídeo")
print("\t-- Tiempo de entrenamiento: " + str(end-start))

# TEST 
start = time.time()
res = seg.segmenta(data[te])
end = time.time()
print("\t-- Tiempo de segmentación: " + str(end-start))
acierto = (np.sum(res == labels[te])+0.0) / labels.shape[0]
print("\t-- Tasa de acierto: " + str(acierto))
#################################################