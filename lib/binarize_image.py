# coding=UTF-8
import numpy as np
import cv2

from analysis import CONT_THRES

# Binariza una imagen pintando de blanco los píxeles del icono y en negro
# todo lo demás
def binarize(labels_seg):
    # Creo una imagen en negro
    img = np.zeros((labels_seg.shape[0],labels_seg.shape[1]))
    # Hallo los píxeles del icono
    mark = (labels_seg==2).astype(np.uint8)*255
    # Hallo los cierres convexos
    contList, hier = cv2.findContours(mark,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contList = [cont for cont in contList if len(cont) > CONT_THRES]
    if len(contList) > 0:
            # Elijo el cierre con mayor área
            cont = contList.index( max(contList, key=lambda x : cv2.contourArea(x[0])) )
            # Pinto el cierre elegido
            cv2.drawContours(img, contList, cont, (255,255,255),cv2.cv.CV_FILLED)
            return img, contList[cont]
    else:
            return img, None