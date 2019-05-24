# coding=UTF-8
import cv2
from scipy.misc import imread
import numpy as np
import os, glob

TR_IMG_PATH = "resources/imgs/train/"
PAINT_IMG_PATH = "resources/imgs/paint/"

def get_tr_img():
    """
    Devuelve dos arrays Numpy con los datos y clases de los píxeles de las
    imágenes de entrenamiento del segmentador.

    Devuelve:
        numpy.ndarray: valores RGB normalizados de los píxeles.
        numpy.ndarray: clases de los datos (0: fondo, 1: línea, 2: marca).
    """
    # Leo las imagenes de entrenamiento
    # tr_img = [cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in glob.glob(TR_IMG_PATH+ "/*.png")]
    # paint_img = [cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in glob.glob(PAINT_IMG_PATH+ "/*.png")]
    tr_img_names = sorted([file for file in glob.glob(os.path.join(TR_IMG_PATH, '*.png'))])
    paint_img_names = sorted([file for file in glob.glob(os.path.join(PAINT_IMG_PATH, '*.png'))])
    tr_img = [imread(name) for name in tr_img_names]
    paint_img = [imread(name) for name in paint_img_names]
    data = None
    labels = None

    for i in range(len(tr_img)):
        data_marca = tr_img[i][np.where(np.all(np.equal(paint_img[i],(255,0,0)),2))]
        data_fondo = tr_img[i][np.where(np.all(np.equal(paint_img[i],(0,255,0)),2))]
        data_linea = tr_img[i][np.where(np.all(np.equal(paint_img[i],(0,0,255)),2))]

        labels_marca = np.zeros(data_marca.shape[0],np.int8) + 2
        labels_fondo = np.zeros(data_fondo.shape[0],np.int8)
        labels_linea = np.ones(data_linea.shape[0],np.int8)

        if data is None:
            data = np.concatenate([data_marca, data_fondo, data_linea])
            # data = ((data+0.0) / np.sum(data,1)[:,np.newaxis])[:,:2]
            labels = np.concatenate([labels_marca,labels_fondo, labels_linea])
        else:
            norm = np.concatenate([data_marca, data_fondo, data_linea])
            # norm = ((norm+0.0) / np.sum(norm,1)[:,np.newaxis])[:,:2]
            data = np.concatenate([data, norm])
            labels = np.concatenate([labels, labels_marca,labels_fondo, labels_linea])
    data = ((data+0.0) / np.sum(data,1)[:,np.newaxis])[:,:2]
    return data, labels