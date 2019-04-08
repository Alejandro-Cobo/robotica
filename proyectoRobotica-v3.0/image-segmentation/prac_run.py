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
import random

def dist(p,q):
    return (p[0]-q[0])**2 + (p[1]-q[1])**2

def midPoint(p,q):
    return [(p[0]+q[0])/2, (p[1]+q[1])/2]

def sarea(a,b,c):
    return 0.5*((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))

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

# fourcc = cv2.cv.CV_FOURCC(*'XVID')
# out = cv2.VideoWriter('videos/analisis.avi', fourcc, 24, (320,240), True)

# Ahora clasifico el video
im_count = 0
times = []
ultimoPSalida = None
ultimaEntrada = None
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

    # Segemtno solo una parte de la imagen
    imDraw = img[80:,:,:]

    # La pongo en formato numpy
    imNp = cv2.cvtColor(imDraw, cv2.COLOR_BGR2RGB)

    # Segmntación
    # Compute rgb normalization
    imNp = np.rollaxis((np.rollaxis(imNp,2)+0.0)/np.sum(imNp,2),0,3)[:,:,:2]
    # Aplico un filtro gaussiano
    imNp = cv2.GaussianBlur(imNp, (0,0), 1)
    # Adapto la imagen al formato de entrada del segmentador
    im2D = np.reshape(imNp, (imNp.shape[0]*imNp.shape[1],imNp.shape[2]))
    # Segmento la imagen
    labels_seg = np.reshape(seg.segmenta(im2D), (imDraw.shape[0], imDraw.shape[1]))
    
    # Hallo los contornos del fondo ignorando las marcas
    backImg = (labels_seg!=1).astype(np.uint8)*255
    contList, hier = cv2.findContours(backImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contList = [cont for cont in contList if len(cont) > 100]
    cv2.drawContours(imDraw, contList, -1, (0,0,255))
    # Número de agujeros
    nHoles = len(contList)
    enCruce = nHoles > 2

    # Busco la flecha si estoy en un cruce
    pSalida = None
    if enCruce:
        # Hallo los contornos de la flecha
        markImg = (labels_seg==2).astype(np.uint8)*255
        contList, hier = cv2.findContours(markImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        contList = [cont for cont in contList if len(cont) > 100]
        if len(contList) > 0:
            cont = contList[0]
            cv2.drawContours(imDraw, contList, -1, (255,0,0))
            # Hallo la elipse
            rect = cv2.fitEllipse(cont)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            # Calculo los puntos que definen el cuadrado 
            # sobre el que se inscribe la elipse
            p1 = (box[0] + box[3]) / 2
            p2 = (box[1] + box[2]) / 2
            p3 = (box[0] + box[1]) / 2
            p4 = (box[2] + box[3]) / 2
            # Calculo los puntos que están situados en los bordes de la imagen
            # según la dirección de la flecha (vector v)
            v = np.array(p2 -p1)
            if all(v != 0):
                # Al principio supongo que pSalida1 está en el borde derecho y pSalida2 en el izquierdo
                pSalida1 = [imDraw.shape[1]-2, p2[1] + ((imDraw.shape[1]-2-p2[0])*v[1])/(v[0])]
                pSalida2 = [0, p1[1] + ((0-p1[0])*v[1])/(v[0])]
                # Los corrijo si se salen de los bordes de la imagen
                if pSalida1[1] < 0:
                    pSalida1 = [p2[0] + ((0-p2[1])*v[0])/(v[1]), 0]
                elif pSalida1[1] > imDraw.shape[0] - 2:
                    pSalida1 = [p2[0] + ((imDraw.shape[0] - 2-p2[1])*v[0])/(v[1]), imDraw.shape[0] - 2]
                if pSalida2[1] < 0:
                    pSalida2 = [p1[0] + ((0-2-p1[1])*v[0])/(v[1]), 0]
                elif pSalida2[1] > imDraw.shape[0] - 2:
                    pSalida2 = [p1[0] + ((imDraw.shape[0]-2-p1[1])*v[0])/(v[1]), imDraw.shape[0] - 2]

                # Estimo la orientación de la flecha según qué mitad tenga más área
                markPts = np.argwhere(labels_seg == 2)
                markPts = [ [pt[1],pt[0]] for pt in markPts]
                mark1 = []
                mark2 = []
                for pt in markPts:
                    sa = sarea(p3,p4,pt)
                    if sa < 0:
                        mark1.append(pt)
                    elif sa > 0:
                        mark2.append(pt)

                if len(mark1) > len(mark2):
                    pSalida = pSalida1
                else:
                    pSalida = pSalida2
                if ultimoPSalida and dist(pSalida, ultimoPSalida) > 200:
                    pSalida = ultimoPSalida
                # Pinto la línea que sale de la flecha y llega al punto de salida
                # cv2.line(imDraw,tuple((box[0] + box[2]) / 2),tuple(pSalida),(0,0,255),2)
                ultimoPSalida = pSalida
    else:
        ultimoPSalida = None
    # Hallo los puntos de la línea en el borde de la imagen
    linImg = (labels_seg==1).astype(np.uint8)*255
    contList, hier = cv2.findContours(linImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contList = [cont for cont in contList if len(cont) > 100]
    bordes = []
    for cont in contList:
        found = False
        for pt in cont:
            pt = pt[0]
            if pt[0] == 1 or pt[0] == imDraw.shape[1]-2 or pt[1] == 1 or pt[1] == imDraw.shape[0]-2:
                if found:
                    bordes[-1].append(pt)
                else:
                    found = True
                    bordes.append([pt])
            else:
                found = False
    bordes = [ borde for borde in bordes if len(borde) > 10 ]
    # Determino la entrada de la línea
    yMax = [-1,-1]
    for i in range(len(bordes)):
        pt = max(bordes[i], key=lambda x : x[1])
        if pt[1] > yMax[1]:
            yMax = pt
            entrada = i
        elif pt[1] == yMax[1] and abs(pt[0]-imDraw.shape[1]/2) < abs(yMax[0]-imDraw.shape[1]/2):
            yMax = pt
            entrada = i
    # Pinto la entrada en verde
    pIn = bordes[entrada][len(bordes[entrada])/2]
    [ cv2.circle(imDraw,tuple(pt),2,(0,255,0),1) for pt in bordes[entrada] ]
    # Determino la salida de la línea
    if not enCruce and (len(bordes)==2):
        salida = (entrada+1)%2
        # Pinto la salida en rojo
        [ cv2.circle(imDraw,tuple(pt),2,(0,0,255),1) for pt in bordes[salida] ]
        pOut = bordes[salida][len(bordes[salida])/2]
        cv2.line(imDraw,tuple(pIn),tuple(pOut),(0,0,255,),2)
    elif enCruce and pSalida:
        minDist = -1
        for i in range(len(bordes)):
            pt = min(bordes[i],key=lambda x : dist(x,pSalida))
            d = dist(pt, pSalida)
            if minDist==-1 or d < minDist:
                salida = i
                minDist = d
        # Pinto la salida en rojo
        [ cv2.circle(imDraw,tuple(pt),2,(0,0,255),1) for pt in bordes[salida] ]
        pOut = bordes[salida][len(bordes[salida])/2]
        cv2.line(imDraw,tuple(pIn),tuple(pOut),(0,0,255,),2)

    # genero la paleta de colores
    paleta = np.array([[0,0,0],[0,0,255],[255,0,0]],dtype=np.uint8)
    # ahora pinto la imagen
    imSeg = cv2.cvtColor(paleta[labels_seg],cv2.COLOR_RGB2BGR)
    cv2.imshow("Imagen procesada", img)

    # Pulsar Espaco para detener el vídeo o 'q' para terminar la ejecución 
    k = cv2.waitKey(1)
    if k == ord(' '):
        cv2.putText(img, "Pausado en el fotograma " + str(im_count), (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
        cv2.imshow("Imagen procesada", img)
        k = cv2.waitKey(0)
    if k == ord('q'):
        break

    # Guardo esta imagen para luego con todas ellas generar un video
    # cv2.imwrite("frames/frame%02d.jpg" % im_count, cv2.cvtColor(paleta[labels_seg], cv2.COLOR_BGR2RGB))
    
    # Guardo el vídeo mostrado por pantalla directamente
    # out.write(img)

    times.append((time.time() - start))

print("Tiempo medio de procesado de una imagen: " + str(np.mean(times)) + " s.")
# out.release()
capture.release()
cv2.destroyAllWindows()

"""
###################### BASURERO ######################
cont = max(contList, key=lambda x : len(x))
    
    # Hallo los cierre convexo
    chull = cv2.convexHull(cont,returnPoints=False)

    # Hallo los agujeros en los cierres convexos
    
    convDef = cv2.convexityDefects(cont, chull)
    listConvDefs = convDef[:,0,:].tolist()
    convDefsLarge = [[init,end,mid,length] for init,end,mid,length in listConvDefs if length>2000]
    if convDefsLarge == None:
            convDefsLarge = []

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
    else:
        # Estimo la orientación de la flecha
        flechaImg = (labels_seg==2).astype(np.uint8)*255
        contList, hier = cv2.findContours(flechaImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        if len(contList) > 0:
            cont = max(contList, key=lambda x : len(x))
            cv2.drawContours(img, cont, -1, (255,0,0))
            
            # rect = cv2.minAreaRect(cont)
            # box = cv2.cv.BoxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(img,[box],0,(255,0,0),2)
    rect = cv2.minAreaRect(cont)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(img,[box],0,(0,0,255),2)
"""

