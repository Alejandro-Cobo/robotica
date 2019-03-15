# coding=UTF-8
import cv, cv2
from scipy.misc import imread, imsave
import numpy as np

import time

import classif as cl

def get_data_labels(im_filename, mark_filename):
    img = imread(im_filename)
    markImg = imread(mark_filename)

    data_marca=img[np.where(np.all(np.equal(markImg,(255,0,0)),2))]
    data_fondo=img[np.where(np.all(np.equal(markImg,(0,255,0)),2))]
    data_linea=img[np.where(np.all(np.equal(markImg,(0,0,255)),2))]

    labels_marca = np.zeros(data_marca.shape[0],np.int8) + 2
    labels_fondo = np.zeros(data_fondo.shape[0],np.int8)
    labels_linea = np.ones(data_linea.shape[0],np.int8)

    data = np.concatenate([data_marca, data_fondo, data_linea])
    data = ((data+0.0) / np.sum(data,1)[:,np.newaxis])[:,:2]
    labels = np.concatenate([labels_marca,labels_fondo, labels_linea])
    return data, labels

# Se crean los datasets de entrenamiento y test
dataTr, labelsTr = get_data_labels("imgs/linea.png", "imgs/lineaMarcada.png")
dataTe, labelsTe = get_data_labels("imgs/linea_test.png","imgs/lineaMarcada_test.png")

############# CLASIFICADOR EUCLÍDEO #############
# ENTRENAMIENTO
start = time.time()
seg = cl.segEuclid(dataTr, labelsTr)
end = time.time()
print("- Clasificador euclídeo")
print("\t-- Tiempo de entrenamiento: " + str((end-start)))

# TEST 
start = time.time()
res = seg.segmenta(dataTe)
end = time.time()
print("\t-- Tiempo de segmentación: " + str((end-start)))
acierto = (np.sum(res == labelsTe)+0.0) / labelsTe.shape[0]
print("\t-- Tasa de acierto: " + str(acierto))


############# CLASIFICADOR KNN #############
# ENTRENAMIENTO
start = time.time()
seg = cl.segKNN(dataTr, labelsTr)
end = time.time()
print("- Clasificador KNN")
print("\t-- Tiempo de entrenamiento: " + str((end-start)))

# TEST 
start = time.time()
res = seg.segmenta(dataTe)
end = time.time()
print("\t-- Tiempo de segmentación: " + str((end-start)))
acierto = (np.sum(res == labelsTe)+0.0) / labelsTe.shape[0]
print("\t-- Tasa de acierto: " + str(acierto))
