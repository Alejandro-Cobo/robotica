import numpy as np
import cv2

def binarize(img, labels_seg):
    mark = (labels_seg==2).astype(np.uint8)*255
    contList, hier = cv2.findContours(mark,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    if len(contList) > 0:
            cont = contList.index( max(contList, key=lambda x : len(x[0])) )
            if len(contList[cont]) < 100:
                return None
            img = np.zeros((img.shape[0],img.shape[1]))
            cv2.drawContours(img, contList, cont, (255,255,255),cv2.cv.CV_FILLED)
            return img
    else:
            return None