import numpy as np
import cv2
import glob
import sklearn.neighbors
import sklearn.cross_validation as cv

def get_hu_moments():
    folders = ["cruz","escaleras","servicio","telefono"]
    n_imgs = 100
    data = np.empty((len(folders)*n_imgs,7))
    labels = np.empty(len(folders)*n_imgs)

    for i in range(len(folders)):
        images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in glob.glob("resources/dataset/imgs/"+folders[i]+"/*.jpg")]
        moments = [cv2.moments(image,True) for image in images]
        hu_moments = np.array([cv2.HuMoments(moment)[0] for moment in moments])
        data[i*n_imgs:(i+1)*n_imgs] = hu_moments
        labels[i*n_imgs:(i+1)*n_imgs] = i

    return data, labels

