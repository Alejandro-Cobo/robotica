# coding=UTF-8
# Librerías externas
import cv2
import numpy as np

# Librerías internas
import geometry as geo

# Constantes numéricas
CONT_THRESH = 100
BORD_THRESH = 5
DIST_THRESH = 10000

def es_cruce(im, labels_seg):
    """
    Devuelve True si hay un cruce o bifurcación y False en otro caso.

    Argumentos:
        im (numpy.ndarray): imagen de entrada.
        labels_seg (numpy.ndarray): imagen segmentada.

    Devuelve:
        bool: True si la imagen representa una bifurcación, False en otro caso.
    """
    # Hallo los contornos del fondo ignorando las marcas
    backImg = (labels_seg!=1).astype(np.uint8)*255
    _, contList, _ = cv2.findContours(backImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contList = [cont for cont in contList if len(cont) > CONT_THRESH]
    # Visualizar los contornos
    # cv2.drawContours(im, contList, -1, (0,0,255))
    # Número de agujeros
    nHoles = len(contList)
    return nHoles > 2

def get_pt_flecha(im, labels_seg, ultimo_pt_flecha):
    """
    Calcula el píxel del borde de la imagen al que apunta la flecha.
    
    Argumentos:
        im (numpy.ndarray): imagen de entrada.
        labels_seg (numpy.ndarray): imagen segmentada.
        ultimo_pt_flecha (list): píxel al que apuntaba la flecha en el frame anterior.
            None si no había ninguno.

    Devuelve:
        list: Posición del píxel de los márgenes de la imagen al que apunta la flecha.
    """
    pt_flecha = None
    # Hallo los contornos de la flecha
    markImg = (labels_seg==2).astype(np.uint8)*255
    _, contList, _ = cv2.findContours(markImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contList = [cont for cont in contList if len(cont) > CONT_THRESH]
    if len(contList) > 0:
        cont = max(contList, key=lambda x : len(x))
        # Visualizar los contornos de la flecha
        # cv2.drawContours(im, cont, -1, (255,0,0))
        # Hallo la elipse
        rect = cv2.fitEllipse(cont)
        box = cv2.boxPoints(rect)
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
        v = np.array(p2-p1)
        if all(v != 0):
            # Al principio supongo que pt_flecha1 está en el borde derecho y pt_flecha2 en el izquierdo
            pt_flecha1 = [im.shape[1]-2, p2[1] + ((im.shape[1]-2-p2[0])*v[1])/(v[0])]
            pt_flecha2 = [0, p1[1] + ((0-p1[0])*v[1])/(v[0])]
            # Los corrijo si se salen de los bordes de la imagen
            if pt_flecha1[1] < 0:
                pt_flecha1 = [p2[0] + ((0-p2[1])*v[0])/(v[1]), 0]
            elif pt_flecha1[1] > im.shape[0] - 2:
                pt_flecha1 = [p2[0] + ((im.shape[0] - 2-p2[1])*v[0])/(v[1]), im.shape[0] - 2]
            if pt_flecha2[1] < 0:
                pt_flecha2 = [p1[0] + ((0-2-p1[1])*v[0])/(v[1]), 0]
            elif pt_flecha2[1] > im.shape[0] - 2:
                pt_flecha2 = [p1[0] + ((im.shape[0]-2-p1[1])*v[0])/(v[1]), im.shape[0] - 2]
            # Visualizar la línea que une ambos puntos
            # cv2.line(im,tuple(pt_flecha1),tuple(pt_flecha2),(0,0,255))

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
                pt_flecha = pt_flecha1
            else:
                pt_flecha = pt_flecha2

            if ultimo_pt_flecha is not None and geo.dist(pt_flecha, ultimo_pt_flecha) > DIST_THRESH:
                assert len(ultimo_pt_flecha) == 2, "len(ultimo_pt_flecha) != 2"
                pt_flecha = ultimo_pt_flecha
            # Visualizar la línea que indica la orientación de la flecha
            # cv2.line(im,tuple((box[0] + box[2]) / 2),tuple(pt_flecha),(255,0,0),1)
        else:
            # TODO: calcular los puntos de salida cuando el vector v es vertical
            pt_flecha = ultimo_pt_flecha
    return pt_flecha

def get_bordes(im, labels_seg):
    """
    Devuelve los píxeles del contorno de la línea que se encuentran 
    en los bordes de la imagen.

    Argumentos:
        im (numpy.ndarray): imagen de entrada.
        labels_seg (numpy.ndarray): imagen segmentada.

    Devuelve:
        list: posiciones de los píxeles de la línea que están sobre los márgenes
            de la imagen.
    """
    linImg = (labels_seg==1).astype(np.uint8)*255
    _, contList, _ = cv2.findContours(linImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contList = [cont for cont in contList if len(cont) > CONT_THRESH]
    # Visualizar el contorno de la línea
    # cv2.drawContours(im, contList,-1,(255,0,0),1)
    bordes = []
    for cont in contList:
        found = False
        for pt in cont:
            pt = pt[0]
            if pt[0] == 0 or pt[0] == im.shape[1]-1 or pt[1] == 0 or pt[1] == im.shape[0]-1:
                # Visualizar los bordes
                # cv2.circle(im,tuple(pt),2,(255,0,0))
                if found:
                    bordes[-1].append(pt)
                else:
                    found = True
                    bordes.append([pt])

            else:
                found = False
    bordes = [ borde for borde in bordes if len(borde) > BORD_THRESH ]

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

def get_entrada(im, bordes, ultimaEntrada):
    """
    Devuelve el índice de la lista de bordes que representan
    la entrada de la línea dada una lista de bordes.

    Argumentos:
        im (numpy.ndarray): imagen de entrada.
        bordes (list): lista de puntos con las posiciones de los píxeles
            que se encuentran en lso márgenes de la imagen.
        ultimaEntrada (list): posición del punto en la mitad de la lista de
            píxeles en el borde de entrada del frame anterior.
    
    Devuelve:
        int: índice de la lista de bordes que indica el borde de entrada, o -1 si
            no existe.

    """
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
    if entrada == -1 and ultimaEntrada is not None:
        return _get_closest_border(bordes, p)
    return entrada

def get_salida(bordes, entrada, pt_flecha, ultima_salida):
    """
    Devuelve el índice de la lista de bordes que representan
    la salida de la línea dada una lista de bordes.

    Argumentos:
        bordes (list): lista de puntos con las posiciones de los píxeles
            que se encuentran en lso márgenes de la imagen.
        pt_flecha (list): posición del píxel en el margen de la imagen al que apunta
            la flecha, o None si no hay.
        ultima_salida (list): posición del punto en la mitad de la lista de
            píxeles en el borde de salida del frame anterior.

    Devuelve:
        int: índice de la lista de bordes que indica el borde de salida, o -1 si
            no existe.
    """
    salida = -1
    # Si solo hay 2 bordes, duevuele el que no es la entrada
    if (len(bordes)==2):
        salida =  (entrada+1)%2
    # Si la flecha marca un punto, asigna la salida al borde que esté más cerca
    elif pt_flecha is not None:
        salida = _get_closest_border(bordes, pt_flecha)
    # Si no hay flecha, se estima el borde eligiendo el que esté más cerca del último
    elif ultima_salida is not None:
        salida = _get_closest_border(bordes, ultima_salida)
    return salida

def _get_closest_border(bordes, p):
    """
    Devuelve el índice del borde de la lista de bordes más cercano al punto p.
    """
    minDist = -1
    for i in range(len(bordes)):
        pt = bordes[i][len(bordes[i])/2]
        d = geo.dist(pt, p)
        if minDist == -1 or d < minDist:
            salida = i
            minDist = d
    return salida