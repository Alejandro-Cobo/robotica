# coding=UTF-8
import numpy as np
import cv2
import glob

# Devuelve la base de datos de imágenes convertida a momentos de Hu
def get_db_hu():
    folders = ["cruz","escaleras","servicio","telefono"]
    n_imgs = 100
    data = np.empty((len(folders)*n_imgs,7))
    labels = np.empty(len(folders)*n_imgs)

    for i in range(len(folders)):
        images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in glob.glob("resources/dataset/imgs/"+folders[i]+"/*.jpg")]
        moments = [cv2.moments(image,True) for image in images]
        hu_moments = np.array([cv2.HuMoments(moment).T[0] for moment in moments])
        data[i*n_imgs:(i+1)*n_imgs] = hu_moments
        labels[i*n_imgs:(i+1)*n_imgs] = i

    return data, labels

# Convierte una imagen a momentos de Hu
def get_hu(img):
    moments = cv2.moments(img, True)
    return cv2.HuMoments(moments).T

