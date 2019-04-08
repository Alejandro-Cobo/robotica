# coding=UTF-8
# Librerías externas
import cv, cv2
import numpy as np

# Librerías internas
import geometry as geo

# Devuelve True si hay un cruce o bifurcación y False en otro caso
def esCruce(im,labels_seg):
    # Hallo los contornos del fondo ignorando las marcas
    backImg = (labels_seg!=1).astype(np.uint8)*255
    contList, hier = cv2.findContours(backImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contList = [cont for cont in contList if len(cont) > 100]
    # Visualizar los contornos
    cv2.drawContours(im, contList, -1, (0,0,255))
    # Número de agujeros
    nHoles = len(contList)
    return nHoles > 2

# Devuelve el píxel del borde de la imagen al que apunta la flecha
def get_pSalida(im, labels_seg, ultimoPSalida):
    pSalida = None
    # Hallo los contornos de la flecha
    markImg = (labels_seg==2).astype(np.uint8)*255
    contList, hier = cv2.findContours(markImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contList = [cont for cont in contList if len(cont) > 100]
    if len(contList) > 0:
        cont = contList[0]
        # Visualizar los contornos
        cv2.drawContours(im, contList, -1, (255,0,0))
        # Hallo la elipse
        rect = cv2.fitEllipse(cont)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        # Calculo los puntos que definen el rombo 
        # sobre el que se inscribe la elipse
        p1 = (box[0] + box[3]) / 2
        p2 = (box[1] + box[2]) / 2
        p3 = (box[0] + box[1]) / 2
        p4 = (box[2] + box[3]) / 2
        # Visualizar los puntos
        # cv2.circle(im,tuple(p1),2,(255,0,0),2)
        # cv2.circle(im,tuple(p2),2,(255,0,0),2)
        # cv2.circle(im,tuple(p3),2,(0,255,0),2)
        # cv2.circle(im,tuple(p4),2,(0,255,0),2)

        # Calculo los puntos que están situados en los bordes de la imagen
        # según la dirección de la flecha (vector v)
        v = np.array(p2 -p1)
        if all(v != 0):
            # Al principio supongo que pSalida1 está en el borde derecho y pSalida2 en el izquierdo
            pSalida1 = [im.shape[1]-2, p2[1] + ((im.shape[1]-2-p2[0])*v[1])/(v[0])]
            pSalida2 = [0, p1[1] + ((0-p1[0])*v[1])/(v[0])]
            # Visualizar la línea que une ambos puntos
            # cv2.line(im,tuple(pSalida1),tuple(pSalida2),(0,0,255))
            # Los corrijo si se salen de los bordes de la imagen
            if pSalida1[1] < 0:
                pSalida1 = [p2[0] + ((0-p2[1])*v[0])/(v[1]), 0]
            elif pSalida1[1] > im.shape[0] - 2:
                pSalida1 = [p2[0] + ((im.shape[0] - 2-p2[1])*v[0])/(v[1]), im.shape[0] - 2]
            if pSalida2[1] < 0:
                pSalida2 = [p1[0] + ((0-2-p1[1])*v[0])/(v[1]), 0]
            elif pSalida2[1] > im.shape[0] - 2:
                pSalida2 = [p1[0] + ((im.shape[0]-2-p1[1])*v[0])/(v[1]), im.shape[0] - 2]

            # Estimo la orientación de la flecha según qué mitad tenga más área
            markPts = np.argwhere(labels_seg == 2)
            markPts = [ [pt[1],pt[0]] for pt in markPts]
            mark1 = []
            mark2 = []
            for pt in markPts:
                sa = geo.sarea(p3,p4,pt)
                if sa < 0:
                    mark1.append(pt)
                elif sa > 0:
                    mark2.append(pt)

            # Visualizar loas mitades de la flecha
            # [ cv2.circle(im,tuple(pt),1,(255,0,0)) for pt in mark1 ]
            # [ cv2.circle(im,tuple(pt),1,(0,255,0)) for pt in mark2 ]
            if len(mark1) > len(mark2):
                pSalida = pSalida1
            else:
                pSalida = pSalida2
            if ultimoPSalida and geo.dist(pSalida, ultimoPSalida) > 200:
                pSalida = ultimoPSalida
            # Pinto la línea que sale de la flecha y llega al punto de salida
            # cv2.line(im,tuple((box[0] + box[2]) / 2),tuple(pSalida),(0,0,255),2)
            ultimoPSalida = pSalida
    return pSalida,ultimoPSalida

# Devuelve los píxeles del contorno de la línea que se encuentran 
# en los bordes de la imagen
def get_bordes(im, labels_seg):
    # Hallo los puntos de la línea en el borde de la imagen
    linImg = (labels_seg==1).astype(np.uint8)*255
    contList, hier = cv2.findContours(linImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contList = [cont for cont in contList if len(cont) > 100]
    bordes = []
    for cont in contList:
        found = False
        for pt in cont:
            pt = pt[0]
            if pt[0] == 1 or pt[0] == im.shape[1]-2 or pt[1] == 1 or pt[1] == im.shape[0]-2:
                # Visualizar los contornos
                # cv2.circle(im,tuple(pt),2,(255,0,0))
                if found:
                    bordes[-1].append(pt)
                else:
                    found = True
                    bordes.append([pt])
            else:
                found = False

    return [ borde for borde in bordes if len(borde) > 5 ]

# Devuelve el índice de la lista de bordes que repredsentan
# la entrada de lal ínea dado una lista de bordes
def get_entrada(im,bordes):
    yMax = [-1,-1]
    for i in range(len(bordes)):
        pt = max(bordes[i], key=lambda x : x[1])
        if pt[1] > yMax[1]:
            yMax = pt
            entrada = i
        elif pt[1] == yMax[1] and abs(pt[0]-im.shape[1]/2) < abs(yMax[0]-im.shape[1]/2):
            yMax = pt
            entrada = i
    return entrada

# Devuelve el índice de la lista de bordes que repredsentan
# la salida de lal ínea dado una lista de bordes.
# Devuelve -1 si está en un cruce y no hay punto de salida
def get_salida(bordes,entrada,pSalida):
    if (len(bordes)==2):
        return (entrada+1)%2
    elif pSalida:
        minDist = -1
        for i in range(len(bordes)):
            pt = min(bordes[i],key=lambda x : geo.dist(x,pSalida))
            d = geo.dist(pt, pSalida)
            if minDist==-1 or d < minDist:
                salida = i
                minDist = d
        return salida
    return -1