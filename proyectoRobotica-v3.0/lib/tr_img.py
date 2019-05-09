# coding=UTF-8
import cv2
from scipy.misc import imread
import numpy as np

def get_tr_img():
    # Leo las imagenes de entrenamiento
    trImg = imread('resources/imgs/tr_img.png')
    trImgPaint = imread('resources/imgs/tr_img_paint.png')

    # Saco todos los puntos marcados en rojo/verde/azul
    data_marca = trImg[np.where(np.all(np.equal(trImgPaint,(255,0,0)),2))]
    data_fondo = trImg[np.where(np.all(np.equal(trImgPaint,(0,255,0)),2))]
    data_linea = trImg[np.where(np.all(np.equal(trImgPaint,(0,0,255)),2))]

    labels_marca = np.zeros(data_marca.shape[0],np.int8) + 2
    labels_fondo = np.zeros(data_fondo.shape[0],np.int8)
    labels_linea = np.ones(data_linea.shape[0],np.int8)

    data = np.concatenate([data_marca, data_fondo, data_linea])
    data = ((data+0.0) / np.sum(data,1)[:,np.newaxis])[:,:2]
    labels = np.concatenate([labels_marca,labels_fondo, labels_linea])

    return data, labels