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
    cv2.waitKey(1)

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

    linImg = (labels_seg==1).astype(np.uint8)*255
    contList, hier = cv2.findContours(linImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, contList, -1, (0,0,255))
    
    chullList = [cv2.convexHull(cont,returnPoints=False) for cont in contList]

    convDefs = [cv2.convexityDefects(cont, chull) for (cont,chull) in zip(contList,chullList) if len(chull) > 3]
    cont = max(contList, key=lambda x : len(x))
    cnvDef = max(convDefs, key=lambda x : x != None and len(x))
    listConvDefs = cnvDef[:,0,:].tolist()
    convDefsLarge = [[init,end,mid,length] for init,end,mid,length in listConvDefs if length>1000]
    escenas = ["Linea recta", "Curva derecha/izquierda", "Cruce con 2 o 3 salidas"]
    # Para pintar texto en una imagen
    if convDefsLarge == None:
            convDefsLarge = []
    cv2.putText(img,'{0}'.format(escenas[min(len(convDefsLarge), 2)]),(15,20),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0))
    # Vuelvo a pintar la imagen
    # genero la paleta de colores
    paleta = np.array([[0,0,0],[0,0,255],[255,0,0]],dtype=np.uint8)
    # ahora pinto la imagen
    imSeg = cv2.cvtColor(paleta[labels_seg],cv2.COLOR_RGB2BGR)
    cv2.imshow("Segmentacion QDA", np.concatenate((img, imSeg), axis=1))

    
    # Para pintar un circulo en el centro de la imagen
    # cv2.circle(imDraw, (imDraw.shape[1]/2,imDraw.shape[0]/2), 2, (0,255,0), -1)

    # Guardo esta imagen para luego con todas ellas generar un video
    # cv2.imwrite("frames/frame%02d.jpg" % im_count, cv2.cvtColor(paleta[labels_seg], cv2.COLOR_BGR2RGB))
    # im_count += 1
    # out.write(cv2.cvtColor(imConcat, cv2.COLOR_BGR2RGB))

    times.append((time.time() - start))

print("Tiempo medio de procesado de una imagen: " + str(np.mean(times)) + " s.")
# out.release()
capture.release()
cv2.destroyAllWindows()

