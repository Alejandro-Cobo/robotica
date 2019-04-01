# coding=UTF-8
####################################################
# Esqueleto de programa para ejecutar el algoritmo de segmentacion.
# Este programa primero entrena el clasificador con los datos de
#  entrenamiento y luego segmenta el video (este entrenamiento podria
#  hacerse en "prac_ent.py" y aqui recuperar los parametros del clasificador
###################################################

# Alejandro Cobo Cabornero, 150333
# Facundo Navarro Olivera, 140213
# Diego Sánchez Lizuain, 150072

import cv, cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import numpy as np

import classif as seg
import time

print("Pulsar Espaco para detener el vídeo o 'q' para terminar la ejecución")

start = time.time()
# Leo las imagenes de entrenamiento
imNp = imread('imgs/linea.png')
markImg = imread('imgs/lineaMarcada.png')

# Saco todos los puntos marcados en rojo/verde/azul
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

fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('videos/analisis.avi', fourcc, 24, (320,240), True)

# Ahora clasifico el video
im_count = 0
times = []
while True:
    start = time.time()

    ret, img = capture.read()
    
    # Segmento una de cada dos imágenes
    im_count += 1
    if im_count % 2 != 0:
        continue
    
    # Si no hay más imágenes termino el bucle
    if not ret:
        break

    # La pongo en formato numpy
    imNp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Segmntación
    # Compute rgb normalization
    imNp = np.rollaxis((np.rollaxis(imNp,2)+0.0)/np.sum(imNp,2),0,3)[:,:,:2]
    # Aplico un filtro gaussiano
    imNp = cv2.GaussianBlur(imNp, (0,0), 1)
    # Adapto la imagen al formato de entrada del segmentador
    im2D = np.reshape(imNp, (imNp.shape[0]*imNp.shape[1],imNp.shape[2]))
    # Segmento la imagen
    labels_seg = np.reshape(seg.segmenta(im2D), (img.shape[0], img.shape[1]))

    # Hallo los contornos
    linImg = (labels_seg==1).astype(np.uint8)*255
    contList, hier = cv2.findContours(linImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contList = [cont for cont in contList if len(cont) > 100]
    cv2.drawContours(img, contList, -1, (0,0,255))
    cont = max(contList, key=lambda x : len(x))
    
    # Hallo los cierre convexo
    chull = cv2.convexHull(cont,returnPoints=False)

    # Hallo los agujeros en los cierres convexos
    convDef = cv2.convexityDefects(cont, chull)
    listConvDefs = convDef[:,0,:].tolist()
    convDefsLarge = [[init,end,mid,length] for init,end,mid,length in listConvDefs if length>2000]
    if convDefsLarge == None:
            convDefsLarge = []

    # Identifico las escenas
    escenas = ["Linea recta", "Curva", "Cruce"]
    text = escenas[min(len(convDefsLarge), 2)]

    # Compruebo si la curva es hacia la derecha o la izquierda
    if len(contList) == 1 and min(len(convDefsLarge), 2) == 1:
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

    # Compruebo el número de salidas del cruce
    elif min(len(convDefsLarge), 2) == 2:
        # Identifico si hay 2 o 3 salidas en función del número de agujeros 
        if len(convDefsLarge) < 4:
            text += " con 2 salidas"
        else:
            text += " con 3 salidas"

    # Pinto la escena identificada en la imagen
    cv2.putText(img, text, (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))    

    # Identifico la entrada y salida de la escena
    maxX = max(cont, key=lambda x : x[0][0])[0]
    minX = min(cont, key=lambda x : [x[0][0], -x[0][1]])[0]
    maxY = max(cont, key=lambda x : x[0][1])[0]
    minY = min(cont, key=lambda x : x[0][1])[0]
    # Líena recta o curva
    if len(contList) == 1 and min(len(convDefsLarge), 2) < 2:  
        # Pinto la entrada y la salida
        for pt in cont:
            if pt[0][1] == img.shape[0] - 2:
                cv2.circle(img, (pt[0][0], pt[0][1]), 2, (0,255,0), -1)
            elif pt[0][1] == 1:
                cv2.circle(img, (pt[0][0], pt[0][1]), 2, (0,0,255), -1)
            elif pt[0][0] == 1:
                if minX[1] > maxX[1]:
                    cv2.circle(img, (pt[0][0], pt[0][1]), 2, (0,255,0), -1)
                else:
                    cv2.circle(img, (pt[0][0], pt[0][1]), 2, (0,0,255), -1)
            elif pt[0][0] == img.shape[1] - 2:
                if maxX[1] > minX[1]:
                    cv2.circle(img, (pt[0][0], pt[0][1]), 2, (0,255,0), -1)
                else:
                    cv2.circle(img, (pt[0][0], pt[0][1]), 2, (0,0,255), -1)

    # Cruce
    """ 
    else:
        # Estimo la orientación de la flecha
        flechaImg = (labels_seg==2).astype(np.uint8)*255
        contList, hier = cv2.findContours(flechaImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        if len(contList) > 0:
            cont = max(contList, key=lambda x : len(x))
            cv2.drawContours(img, cont, -1, (255,0,0))
            rect = cv2.minAreaRect(cont)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(255,0,0),2)
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
    cv2.imshow("Segmentacion QDA", img)

    # Pulsar Espaco para detener el vídeo o 'q' para terminar la ejecución 
    k = cv2.waitKey(1)
    if k == ord(' '):
        k = cv2.waitKey(0)
    if k == ord('q'):
        break

    # Guardo esta imagen para luego con todas ellas generar un video
    # cv2.imwrite("frames/frame%02d.jpg" % im_count, cv2.cvtColor(paleta[labels_seg], cv2.COLOR_BGR2RGB))
    
    # Guardo el vídeo mostrado por pantalla directamente
    out.write(img)

    times.append((time.time() - start))

print("Tiempo medio de procesado de una imagen: " + str(np.mean(times)) + " s.")
out.release()
capture.release()
cv2.destroyAllWindows()

