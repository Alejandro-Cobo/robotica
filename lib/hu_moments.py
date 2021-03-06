# coding=UTF-8
import numpy as np
import cv2
import glob

# Nombre de los directorios de la base de datos
FOLDERS = ["cruz","escaleras","servicio","telefono"]
# Número de imágenes por directorio
N_IMGS = 100

def get_db():
    """
    Devuelve la base de datos de imágenes convertida a momentos de Hu.

    Devuelve:
        numpy.ndarray: matriz con los momentos de Hu.
        numpy.ndarray: lista de las clases de los datos.
    """
    data = np.empty((len(FOLDERS)*N_IMGS,7))
    labels = np.empty(len(FOLDERS)*N_IMGS)

    for i in range(len(FOLDERS)):
        images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in glob.glob("resources/dataset/"+FOLDERS[i]+"/*.jpg")]
        # moments = [cv2.moments(image,True) for image in images]
        # hu_moments = np.array([cv2.HuMoments(moment).T[0] for moment in moments])
        hu_moments = [get_hu(img) for img in images]
        data[i*N_IMGS:(i+1)*N_IMGS] = hu_moments
        labels[i*N_IMGS:(i+1)*N_IMGS] = i

    return data, labels

def get_hu(img):
    """
    Convierte una imagen binarizada a momentos de Hu.
    """
    moments = cv2.moments(img, True)
    return cv2.HuMoments(moments).T
    