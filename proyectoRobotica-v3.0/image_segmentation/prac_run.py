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

# Librerías externas
import cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
import time

# Librerías internas
import classif as seg
import analisis
import geometry as geo
import symbol_recognition.binarize_image as bin
import symbol_recognition.hu_moments as hu
import symbol_recognition.mahalanobis as mahalanobis

print("Pulsar Espacio para detener el vídeo o 'q' para terminar la ejecución")

start = time.time()
# Leo las imagenes de entrenamiento
imNp = imread('resources/imgs/linea3.png')
markImg = imread('resources/imgs/lineaMarcada3.png')

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

# Creo y entreno el reconocedor de símbolos
maha = mahalanobis.classifMahalanobis()
data, labels = hu.get_db_hu()
maha.fit(data, labels)

# Lista de símbolos a reconocer
symbols = ["Cruz", "Escaleras", "Servicio", "Telefono"]

print("Tiempo de entrenamiento: " + str(time.time() - start) + " s.")

# Inicio la captura de imagenes
capture = cv2.VideoCapture("resources/videos/dynamic_test_1.webm")

cv2.namedWindow('Imagen procesada',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Imagen procesada', 720,405)

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
    # img = img[:,:img.shape[1]-80,:]
    imDraw = img

    # La pongo en formato numpy
    imNp = cv2.cvtColor(imDraw, cv2.COLOR_BGR2RGB)

    # Segmntación
    # Compute rgb normalization
    imNp = np.rollaxis((np.rollaxis(imNp,2)+0.0)/np.sum(imNp,2),0,3)[:,:,:2]
    # Aplico un filtro gaussiano
    imNp = cv2.GaussianBlur(imNp, (0,0), 1)
    # Adapto la imagen al formato de entrada del segmentador
    im2D = np.reshape(imNp, (imNp.shape[0]*imNp.shape[1],imNp.shape[2]))
    im2D = np.nan_to_num(im2D)
    # Segmento la imagen
    labels_seg = np.reshape(seg.segmenta(im2D), (imDraw.shape[0], imDraw.shape[1]))


    # Compruebo si estoy en un cruce
    enCruce = analisis.esCruce(imDraw,labels_seg)
    """
    if enCruce:
        cv2.putText(img, "Cruce detectado", (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
    else:
        cv2.putText(img, "Sin cruces", (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
    """
    
    # Busco la flecha si estoy en un cruce
    if enCruce:
        pSalida, ultimoPSalida = analisis.get_pSalida(imDraw, labels_seg, ultimoPSalida)
    else:
        pSalida = None
        ultimoPSalida = None
        bin_img = bin.binarize(img, labels_seg)
        if bin_img is not None:
            hu_moments = hu.get_hu(bin_img)
            symbol = symbols[ int(maha.predict(hu_moments)) ]
            cv2.putText(img, symbol, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
    """
    # Hallo los puntos de la línea en el borde de la imagen
    bordes = analisis.get_bordes(imDraw,labels_seg)

    # Determino la entrada de la línea
    entrada = analisis.get_entrada(imDraw,bordes)
    # Punto medio
    pIn = bordes[entrada][len(bordes[entrada])/2]
    # Pinto los píxeles de entrada en verde
    [ cv2.circle(imDraw,tuple(pt),2,(0,255,0),1) for pt in bordes[entrada] ]
    # Determino la salida de la línea
    salida = analisis.get_salida(bordes,entrada,pSalida)
    if salida != -1:
        # Pinto los píxeles salida en rojo
        [ cv2.circle(imDraw,tuple(pt),2,(0,0,255),1) for pt in bordes[salida] ]
        pOut = bordes[salida][len(bordes[salida])/2]
        # Pinto la líne aque une la entrada y la salida
        cv2.line(imDraw,tuple(pIn),tuple(pOut),(0,0,255,),2)
    """
    # genero la paleta de colores
    # paleta = np.array([[0,0,0],[0,0,255],[255,0,0]],dtype=np.uint8)
    # ahora pinto la imagen
    # imSeg = cv2.cvtColor(paleta[labels_seg],cv2.COLOR_RGB2BGR)
    # cv2.imshow("Imagen segmentada", imSeg)
    cv2.imshow("Imagen procesada", img)
    # Guardo el vídeo mostrado por pantalla
    # out.write(img)

    # Pulsar Espaco para detener el vídeo o 'q' para terminar la ejecución
    
    k = cv2.waitKey(1)
    if k == ord(' '):
        cv2.putText(img, "Pausado en el fotograma " + str(im_count), (10,60), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
        cv2.imshow("Imagen procesada", img)
        k = cv2.waitKey(0)
    if k == ord('q'):
        break

    times.append((time.time() - start))

# out.release()
capture.release()
cv2.destroyAllWindows()
print("Tiempo medio de procesado de una imagen: " + str(np.mean(times)) + " s.")
