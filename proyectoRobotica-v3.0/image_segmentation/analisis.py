# coding=UTF-8
# Librerías externas
import cv2
import numpy as np

# Librerías internas
import geometry as geo

# Constantes numéricas
CONT_THRES = 100
BORD_THRES = 5
DIST_THRES = 50

# Devuelve True si hay un cruce o bifurcación y False en otro caso
def esCruce(im,labels_seg):
    # Hallo los contornos del fondo ignorando las marcas
    backImg = (labels_seg!=1).astype(np.uint8)*255
    contList, hier = cv2.findContours(backImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contList = [cont for cont in contList if len(cont) > CONT_THRES]
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
    contList = [cont for cont in contList if len(cont) > CONT_THRES]
    if len(contList) > 0:
        cont = max(contList, key=lambda x : len(x))
        # Visualizar los contornos de la flecha
        cv2.drawContours(im, cont, -1, (255,0,0))
        # Hallo la elipse
        rect = cv2.fitEllipse(cont)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        # Calculo los puntos que definen los ejes de la elipse
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
            # Los corrijo si se salen de los bordes de la imagen
            if pSalida1[1] < 0:
                pSalida1 = [p2[0] + ((0-p2[1])*v[0])/(v[1]), 0]
            elif pSalida1[1] > im.shape[0] - 2:
                pSalida1 = [p2[0] + ((im.shape[0] - 2-p2[1])*v[0])/(v[1]), im.shape[0] - 2]
            if pSalida2[1] < 0:
                pSalida2 = [p1[0] + ((0-2-p1[1])*v[0])/(v[1]), 0]
            elif pSalida2[1] > im.shape[0] - 2:
                pSalida2 = [p1[0] + ((im.shape[0]-2-p1[1])*v[0])/(v[1]), im.shape[0] - 2]
            # Visualizar la línea que une ambos puntos
            # cv2.line(im,tuple(pSalida1),tuple(pSalida2),(0,0,255))

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

            # Visualizar las mitades de la flecha
            # [ cv2.circle(im,tuple(pt),1,(255,0,0)) for pt in mark1 ]
            # [ cv2.circle(im,tuple(pt),1,(0,255,0)) for pt in mark2 ]
            if len(mark1) > len(mark2):
                pSalida = pSalida1
            else:
                pSalida = pSalida2
            if (ultimoPSalida is not None) and (geo.dist(pSalida, ultimoPSalida) > DIST_THRES):
                pSalida = ultimoPSalida
            # Visualizar la línea que indica la orientación de la flecha
            cv2.line(im,tuple((box[0] + box[2]) / 2),tuple(pSalida),(255,0,0),1)
    return pSalida

# Devuelve los píxeles del contorno de la línea que se encuentran 
# en los bordes de la imagen
def get_bordes(im, labels_seg):
    # Hallo los puntos de la línea en el borde de la imagen
    linImg = (labels_seg==1).astype(np.uint8)*255
    contList, hier = cv2.findContours(linImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contList = [cont for cont in contList if len(cont) > CONT_THRES]
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
    bordes = [ borde for borde in bordes if len(borde) > BORD_THRES ]

    if len(bordes) > 0:
        # Caso particular en el que, si un borde se extiende sobre la 
        # esquina superior izquierda de la imagen, se considera como dos
        # bordes distintos y hay que volver a juntarlos
        xMin = min(bordes[0],key=lambda x : x[1])
        yMin = min(bordes[-1],key = lambda x: x[0])
        if xMin[0] == 1 and xMin[1] == 1 and yMin[0] == 2 and yMin[1] == 1:
            bordes[0] += bordes[-1]
            del bordes[-1]

    return bordes

# Devuelve el índice de la lista de bordes que representan
# la entrada de lal ínea dado una lista de bordes
def get_entrada(im, bordes, ultimaEntrada):
    entrada = -1
    yMax = [-1,-1]
    for i in range(len(bordes)):
        pt = bordes[i][len(bordes[i])/2]
        if pt[1] > yMax[1]:
            yMax = pt
            entrada = i
        # En caso de empate, elegir el borde más cercano al centro
        elif pt[1] == yMax[1] and abs(pt[0]-im.shape[1]/2) < abs(yMax[0]-im.shape[1]/2):
        # Alternativa: elegir el borde más cercano al último borde
        # elif pt[1] == yMax[1] and geo.dist(pt, ultimaEntrada) < geo.dist(yMax, ultimaEntrada):
            yMax = pt
            entrada = i
    return entrada

# Devuelve el índice de la lista de bordes que repredsentan
# la salida de lal ínea dado una lista de bordes.
# Devuelve -1 si está en un cruce y no hay punto de salida
def get_salida(bordes, entrada, pSalida, ultimaSalida):
    salida = -1
    if (len(bordes)==2):
        salida =  (entrada+1)%2
    elif pSalida is not None:
        salida = _get_closest_border(bordes, pSalida)
    elif ultimaSalida is not None:
        salida = _get_closest_border(bordes, ultimaSalida)
    return salida

# Devuelve el índice del borde más cercano al punto p
def _get_closest_border(bordes, p):
    minDist = -1
    for i in range(len(bordes)):
        pt = bordes[i][len(bordes[i])/2]
        d = geo.dist(pt, p)
        if minDist == -1 or d < minDist:
            salida = i
            minDist = d
    return salida