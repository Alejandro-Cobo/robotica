# coding=UTF-8
####################################################
# Esqueleto de programa para ejecutar el algoritmo de segmentacion.
# Este programa primero entrena el clasificador con los datos de
#  entrenamiento y luego segmenta el video (este entrenamiento podria
#  hacerse en "prac_ent.py" y aqui recuperar los parametros del clasificador
###################################################


import cv, cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import numpy as np

import classif as seg
import time

start = time.time()
# Leo las imagenes de entrenamiento
imNp = imread('imgs/linea.png')
markImg = imread('imgs/lineaMarcada.png')

# saco todos los puntos marcados en rojo/verde/azul

data_marca = imNp[np.where(np.all(np.equal(markImg,(255,0,0)),2))]
data_fondo = imNp[np.where(np.all(np.equal(markImg,(0,255,0)),2))]
data_linea = imNp[np.where(np.all(np.equal(markImg,(0,0,255)),2))]

labels_marca = np.zeros(data_marca.shape[0],np.int8) + 2
labels_fondo = np.zeros(data_fondo.shape[0],np.int8)
labels_linea = np.ones(data_linea.shape[0],np.int8)

data = np.concatenate([data_marca, data_fondo, data_linea])
data = ((data+0.0) / np.sum(data,1)[:,np.newaxis])[:,:2]
labels = np.concatenate([labels_marca,labels_fondo, labels_linea])

# Creo y entreno el segmentador
seg = seg.segQDA(data, labels)
print("Tiempo de entrenamiento: " + str(time.time() - start) + " s.")

# Inicio la captura de imagenes
capture = cv2.VideoCapture("videos/video2017-4.avi")

# fourcc = cv2.cv.CV_FOURCC(*'XVID')
# out = cv2.VideoWriter('videos/segmentacion.avi', fourcc, 24, (320*2,240), True)

# Ahora clasifico el video
im_count = 0
times = []
while True:
    start = time.time()

    ret, img = capture.read()

    im_count += 1
    if im_count % 3 != 0:
        continue
        
    if not ret:
        break

    # La pongo en formato numpy
    imNp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Segmento la imagen.
    # Compute rgb normalization
    imNp = np.rollaxis((np.rollaxis(imNp,2)+0.0)/np.sum(imNp,2),0,3)[:,:,:2]
    # Aplico un filtro gaussiano
    imNp = cv2.GaussianBlur(imNp, (0,0), 1)
    im2D = np.reshape(imNp, (imNp.shape[0]*imNp.shape[1],imNp.shape[2]))
    labels_seg = np.reshape(seg.segmenta(im2D), (img.shape[0], img.shape[1]))

    # Contornos
    linImg = (labels_seg==1).astype(np.uint8)*255
    contList, hier = cv2.findContours(linImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cont = max(contList, key=lambda x : len(x))
    cv2.drawContours(img, [cont], -1, (0,0,255))
    
    # Cierre convexo
    chull = cv2.convexHull(cont,returnPoints=False)

    # Agujeros
    convDef = cv2.convexityDefects(cont, chull)
    listConvDefs = convDef[:,0,:].tolist()
    convDefsLarge = [[init,end,mid,length] for init,end,mid,length in listConvDefs if length>2000]

    # Identifico las escenas
    escenas = ["Linea recta", "Curva", "Cruce"]

    if convDefsLarge == None:
            convDefsLarge = []

    text = escenas[min(len(convDefsLarge), 2)]

    # Curva
    if min(len(convDefsLarge), 2) == 1:
        # Identifico si es hacia la izaruierda o la derecha usando el 
        # area signada de los 3 puntos que definen el agujero (init, mid, end)
        init = cont[convDefsLarge[0][0]][0]
        mid = cont[convDefsLarge[0][2]][0]
        end = cont[convDefsLarge[0][1]][0]
        if init[1] < end[1]:
            init, end = end, init
        sarea = 0.5*((mid[0]-end[0])*(init[1]-end[1]) - (init[0]-end[0])*(mid[1]-end[1]))
        if sarea < 0:
            text += " hacia la derecha"
        else:
            text += " hacia la izquierda"

    # Cruce
    elif min(len(convDefsLarge), 2) == 2:
        # Identifico si hay 2 o 3 salidas en función del número de agujeros 
        if len(convDefsLarge) < 4:
            text += " con 2 salidas"
        else:
            text += " con 3 salidas"

    cv2.putText(img, text, (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
    
    """
    # Hallo las entradas y salidas
    maxX = max(cont, key=lambda x : x[0][0])[0]
    maxY = max(cont, key=lambda x : x[0][1])[0]
    minX = min(cont, key=lambda x : x[0][0])[0]
    minY = min(cont, key=lambda x : x[0][1])[0]

    for pt in cont:
        if maxY[1] == img.shape[0] - 2:
            if pt[0][1] == maxY[1]:
                cv2.circle(img, (pt[0][0], pt[0][1]), 2, (0,255,0), -1)
        else:
            
        if minY[1] == 1:
            if pt[0][1] == minY[1]:
                cv2.circle(img, (pt[0][0], pt[0][1]), 2, (0,0,255), -1)
        else:
        """
            

    """
    rect = cv2.minAreaRect(cont)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(img,[box],0,(0,0,255),2)
    """

    # Vuelvo a pintar la imagen
    # genero la paleta de colores
    paleta = np.array([[0,0,0],[0,0,255],[255,0,0]],dtype=np.uint8)
    # ahora pinto la imagen
    imSeg = cv2.cvtColor(paleta[labels_seg],cv2.COLOR_RGB2BGR)
    cv2.imshow("Segmentacion QDA", np.concatenate((img, imSeg), axis=1))
    k = cv2.waitKey(1)
    if k == ord(' '):
        cv2.waitKey(0)
    
    # Para pintar un circulo en el centro de la imagen
    # cv2.circle(imDraw, (imDraw.shape[1]/2,imDraw.shape[0]/2), 2, (0,255,0), -1)

    # Guardo esta imagen para luego con todas ellas generar un video
    # cv2.imwrite("frames/frame%02d.jpg" % im_count, cv2.cvtColor(paleta[labels_seg], cv2.COLOR_BGR2RGB))
    # out.write(cv2.cvtColor(imConcat, cv2.COLOR_BGR2RGB))

    times.append((time.time() - start))

print("Tiempo medio de procesado de una imagen: " + str(np.mean(times)) + " s.")
# out.release()
capture.release()
cv2.destroyAllWindows()

