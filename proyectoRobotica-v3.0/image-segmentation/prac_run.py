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

# Leo las imagenes de entrenamiento
imNp = imread('linea.png')
markImg = imread('lineaMarcada.png')

# saco todos los puntos marcados en rojo/verde/azul
data_marca=imNp[np.where(np.all(np.equal(markImg,(255,0,0)),2))]
data_fondo=imNp[np.where(np.all(np.equal(markImg,(0,255,0)),2))]
data_linea=imNp[np.where(np.all(np.equal(markImg,(0,0,255)),2))]

# Creo y entreno los segmentadores euclideos
start = time.time()
segmEuc = seg.segEuclid([data_fondo, data_linea, data_marca])
# segmMano = seg.segMano([data_fondo, data_linea, data_marca])
end = time.time()
print("Tiempo de entrenamiento: " + str(end-start))
# Inicio la captura de imagenes
capture = cv2.VideoCapture("video1.mp4")
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('video_segmentado.avi', fourcc, 100/25, (320,240))

# Ahora clasifico el video
key = 0
count = 0
im_count = 1
while key != ord('q'):
    # voy a segmentar solo una de cada 25 imagenes y la muestro
    key = cv2.waitKey(1)
    ret, img = capture.read()
    
    if count%25 != 0:
        count += 1
        continue
        
    if not ret:
        break
        
    cv2.imshow("Imagen",img)
    

    # La pongo en formato numpy
    imNp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Segmento la imagen.
    # Compute rgb normalization
    if count == 0:
        start = time.time()
    imrgbn = np.rollaxis((np.rollaxis(imNp,2)+0.0)/np.sum(imNp,2),0,3)[:,:,:2]

    labelsEu=segmEuc.segmenta(imrgbn)
    # labelsMa=segmMano.segmenta(imNp)
    if count == 0:
        end = time.time()
        print("Tiempo de segmentacion: " +str(end-start))

    # Vuelvo a pintar la imagen
    # genero la paleta de colores
    paleta = np.array([[0,0,0],[0,0,255],[255,0,0],[0,255,0]],dtype=np.uint8)
    # ahora pinto la imagen
    cv2.imshow("Segmentacion Euclid",cv2.cvtColor(paleta[labelsEu],cv2.COLOR_RGB2BGR))
    # cv2.imshow("Segmentacion Mano",cv2.cvtColor(paleta[labelsMa],cv2.COLOR_RGB2BGR))

    # Para pintar texto en una imagen
    # cv2.putText(imDraw,'Lineas: {0}'.format(len(convDefsLarge)),(15,20),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0))
    # Para pintar un circulo en el centro de la imagen
    # cv2.circle(imDraw, (imDraw.shape[1]/2,imDraw.shape[0]/2), 2, (0,255,0), -1)

    # Guardo esta imagen para luego con todas ellas generar un video
    # cv2.imwrite("frames/frame%02d.jpg" % im_count, cv2.cvtColor(paleta[labelsEu], cv2.COLOR_BGR2RGB))
    out.write(cv2.cvtColor(paleta[labelsEu], cv2.COLOR_BGR2RGB))
    count += 1
    im_count += 1
    
capture.release()
out.release()
cv2.destroyAllWindows()

