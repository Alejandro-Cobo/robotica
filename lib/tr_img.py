# coding=UTF-8
import cv2
from scipy.misc import imread
import numpy as np

TR_IMG_PATH = "resources/imgs"

def get_tr_img(old = False):
    """
    Devuelve dos arrays Numpy con los datos y clases de los píxeles de las
    imágenes de entrenamiento del segmentador.

    Argumentos:
        old (boolean): si es True, devuelve los datos de las imágenes de entrenamiento
            generadas con los vídeos de otros años. Si es False, devuelve los datos de
            las imágenes del circuito de este año.

    Devuelve:
        numpy.ndarray: valores RGB normalizados de los píxeles.
        numpy.ndarray: clases de los datos (0: fondo, 1: línea, 2: marca).
    """
    # Leo las imagenes de entrenamiento
    if old:
        trImg = imread(TR_IMG_PATH + "/tr_img_old.png")
        trImgPaint = imread( TR_IMG_PATH + "/tr_img_old_paint.png")
    else:
        trImg = imread(TR_IMG_PATH + "/tr_img.png")
        trImgPaint = imread(TR_IMG_PATH + "/tr_img_paint.png")

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