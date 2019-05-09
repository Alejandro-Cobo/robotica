# coding=UTF-8
# Librerías externas
import cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
import time
from sklearn import discriminant_analysis as da

# Librerías internas
from lib import tr_img
from lib import analisis
from lib import geometry as geo
from lib import binarize_image as bin
from lib import hu_moments as hu
from lib import mahalanobis as mahalanobis

print("Pulsar 'Espacio' para detener el vídeo o 'q' para terminar la ejecución")

start = time.time()

# Datos de entrenamiento del segmentador
data, labels = tr_img.get_tr_img()

# Creo y entreno el segmentador
seg = da.QuadraticDiscriminantAnalysis().fit(data, labels)

# Creo y entreno el reconocedor de símbolos
maha = mahalanobis.classifMahalanobis()
data, labels = hu.get_db()
maha.fit(data, labels)

print("Tiempo de entrenamiento: " + str(time.time() - start) + " s.")

# Lista de símbolos a reconocer
symbols = ["Cruz", "Escaleras", "Servicio", "Telefono"]

# Inicio la captura de imagenes
capture = cv2.VideoCapture("resources/videos/dynamic_test_1.mp4")

# Guardar vídeo
# fourcc = cv2.cv.CV_FOURCC(*'XVID')
# out = cv2.VideoWriter('resources/videos/reconocimiento_de_simbolos.avi', fourcc, 30, (352,440), True)

# Ahora clasifico el video
# Contador de frames
im_count = 0
# Lista de tiempos de cada iteración del bucle
times = []
# Último punto indicado por una flecha en un cruce
ultimoPSalida = None
# Punto en la mitad del último contorno de entrada
ultimaEntrada = None
# Punto en la mitad del último contorno de salida
ultimaSalida = None
while True:
    start = time.time()

    ret, img = capture.read()
    
    # Segmento una de cada dos imágenes
    im_count = (im_count + 1) % 2
    if im_count == 0:
        continue

    # Si no hay más imágenes termino el bucle
    if not ret:
        break
    
    # Segemtno solo una parte de la imagen
    imDraw = img[img.shape[0]/4:,:,:]
    
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
    labels_seg = np.reshape(seg.predict(im2D), (imDraw.shape[0], imDraw.shape[1]))
    
    # Compruebo si estoy en un cruce
    enCruce = analisis.esCruce(imDraw,labels_seg)
    
    # Busco la flecha si estoy en un cruce
    if enCruce:
        cv2.putText(img, "Cruce detectado", (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        pSalida = analisis.get_pSalida(imDraw, labels_seg, ultimoPSalida)
        ultimoPSalida = pSalida
    else:
        cv2.putText(img, "Sin cruces", (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        pSalida = None
        ultimoPSalida = None

        # Reconocimiento de símbolos
        # Binarizo la imagen
        imBin, cont = bin.binarize(labels_seg)
        if cont is not None:
            # Visualizar los contornos del símbolo
            cv2.drawContours(imDraw, cont, -1, (255,0,0))
            # Calculo los momentos de Hu
            hu_moments = hu.get_hu(imBin)
            # Clasifico el símbolo con la distancia de Mahalanobis
            symbol = symbols[ int(maha.predict(hu_moments)) ]
            cv2.putText(img, symbol,(10,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))

    # Hallo los puntos de la línea en el borde de la imagen
    bordes = analisis.get_bordes(imDraw,labels_seg)

    # Determino la entrada de la línea
    entrada = analisis.get_entrada(imDraw, bordes, ultimaEntrada)
    if entrada == -1:
        ultimaEntrada = None
        continue
    ultimaEntrada = bordes[entrada][len(bordes[entrada])/2]
    # Punto medio
    pIn = bordes[entrada][len(bordes[entrada])/2]
    # Visualizar los píxeles de entrada en verde
    [ cv2.circle(imDraw,tuple(pt),2,(0,255,0),1) for pt in bordes[entrada] ]
    # Determino la salida de la línea
    salida = analisis.get_salida(bordes, entrada, pSalida, ultimaSalida)
    ultimaSalida = None
    if salida != -1:
        ultimaSalida = bordes[salida][len(bordes[salida])/2]
        # Visualizar los píxeles de salida en rojo
        [ cv2.circle(imDraw,tuple(pt),2,(0,0,255),1) for pt in bordes[salida] ]
        pOut = bordes[salida][len(bordes[salida])/2]
        # Visualizar la línea que une la entrada y la salida
        # cv2.line(imDraw,tuple(pIn),tuple(pOut),(0,0,255,),2)

    # Visualizar la segmentación
    # paleta = np.array([[0,0,0],[0,0,255],[255,0,0]],dtype=np.uint8)
    # imSeg = cv2.cvtColor(paleta[labels_seg],cv2.COLOR_RGB2BGR)
    # cv2.imshow("Imagen segmentada", imSeg)

    # Visualizar la binarización de la imagen
    # cv2.imshow("Imagen binarizada", imBin)

    # Visualizar el análisis
    cv2.imshow("Imagen procesada", img)

    # Guardo el vídeo mostrado por pantalla
    # out.write(img)
    
    # Pulsar Espaco para detener el vídeo o 'q' para terminar la ejecución
    k = cv2.waitKey(1)
    if k == ord(' '):
        cv2.putText(img, "Video pausado", (10,60), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv2.imshow("Imagen procesada", img)
        k = cv2.waitKey(0)
    if k == ord('q'):
        break
    
    times.append((time.time() - start))

# out.release()
capture.release()
cv2.destroyAllWindows()
print("Tiempo medio de procesado de una imagen: " + str(np.mean(times)) + " s.")
