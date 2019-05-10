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
from lib import analysis
from lib import geometry as geo
from lib import binarize_image as bin
from lib import hu_moments as hu
from lib import mahalanobis as mahalanobis

print("Pulsar 'Espacio' para detener el vídeo o 'q' para terminar la ejecución")

start = time.time()

# Datos de entrenamiento del segmentador
data, labels = tr_img.get_tr_img(old=True)

# Creo y entreno el segmentador
seg = da.QuadraticDiscriminantAnalysis().fit(data, labels)

# Creo y entreno el reconocedor de símbolos
maha = mahalanobis.classifMahalanobis()
data, labels = hu.get_db()
maha.fit(data, labels)

tr_time = time.time() - start

# Lista de símbolos a reconocer
symbols = ["Cruz", "Escaleras", "Servicio", "Telefono"]

# Inicio la captura de imágenes
capture = cv2.VideoCapture("resources/videos/video2017-4.avi")

# Guardar vídeo
# fourcc = cv2.cv.CV_FOURCC(*'XVID')
# out = cv2.VideoWriter('resources/videos/reconocimiento_de_simbolos.avi', fourcc, 30, (352,440), True)

# Ahora clasifico el video
# Contador de frames
im_count = 0
# Lista de tiempos de cada iteración del bucle
times_seg = []
times_proc = []
# Punto en la mitad del último contorno de entrada
ultima_entrada = None
# Punto en la mitad del último contorno de salida
ultima_salida = None
# Último punto indicado por la flecha
ultimo_pt_flecha = None
while True:
    ret, img = capture.read()

    # Si no hay más imágenes termino el bucle
    if not ret:
        break
    
    im_count += 1
    
    start = time.time()
    # Segemtno solo una parte de la imagen
    im_draw = img[img.shape[0]/4:,:,:]
    
    # La pongo en formato numpy
    imNp = cv2.cvtColor(im_draw, cv2.COLOR_BGR2RGB)

    # Segmntación
    # Compute rgb normalization
    imNp = np.rollaxis((np.rollaxis(imNp,2)+0.0)/np.sum(imNp,2),0,3)[:,:,:2]
    # Aplico un filtro gaussiano
    imNp = cv2.GaussianBlur(imNp, (0,0), 1)
    # Adapto la imagen al formato de entrada del segmentador
    imNp = np.reshape(imNp, (imNp.shape[0]*imNp.shape[1],imNp.shape[2]))
    imNp = np.nan_to_num(imNp)
    # Segmento la imagen
    labels_seg = np.reshape(seg.predict(imNp), (im_draw.shape[0], im_draw.shape[1]))

    times_seg.append((time.time() - start))
    
    start = time.time()
    # Compruebo si estoy en un cruceq
    en_cruce = analysis.esCruce(im_draw,labels_seg)
    
    # Busco la flecha si estoy en un cruce
    if en_cruce:
        cv2.putText(img, "Cruce detectado", (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        pt_flecha = analysis.get_pt_flecha(im_draw, labels_seg, ultimo_pt_flecha)
        ultimo_pt_flecha = pt_flecha
    else:
        cv2.putText(img, "Sin cruces", (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        pt_flecha = None
        ultimo_pt_flecha = None

        # Reconocimiento de símbolos
        # Binarizo la imagen
        imBin, cont = bin.binarize(labels_seg)
        if cont is not None:
            # Visualizar los contornos del símbolo
            cv2.drawContours(im_draw, cont, -1, (255,0,0))
            # Calculo los momentos de Hu
            hu_moments = hu.get_hu(imBin)
            # Clasifico el símbolo con la distancia de Mahalanobis
            symbol = symbols[ int(maha.predict(hu_moments)) ]
            cv2.putText(img, symbol,(10,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))

    # Hallo los puntos de la línea en el borde de la imagen
    bordes = analysis.get_bordes(im_draw,labels_seg)

    # Determino la entrada de la línea
    entrada = analysis.get_entrada(im_draw, bordes, ultima_entrada)
    # Punto medio
    p_in = bordes[entrada][len(bordes[entrada])/2]
    ultima_entrada = p_in
    # Visualizar los píxeles de entrada en verde
    [ cv2.circle(im_draw,tuple(pt),2,(0,255,0),1) for pt in bordes[entrada] ]
    # Determino la salida de la línea
    salida = analysis.get_salida(bordes, entrada, pt_flecha, ultima_salida)
    ultima_salida = None
    if salida != -1:
        # Visualizar los píxeles de salida en rojo
        [ cv2.circle(im_draw,tuple(pt),2,(0,0,255),1) for pt in bordes[salida] ]
        p_out = bordes[salida][len(bordes[salida])/2]
        ultima_salida = p_out
        # Visualizar la línea que une la entrada y la salida
        # cv2.line(im_draw,tuple(p_in),tuple(p_out),(0,0,255,),2)

    times_proc.append((time.time() - start))

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
        cv2.putText(img, "Video pausado en el frame {}".format(im_count), (10,60), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv2.imshow("Imagen procesada", img)
        k = cv2.waitKey(0)
    if k == ord('q'):
        break

# out.release()
capture.release()
cv2.destroyAllWindows()
print("******************** Informe de tiempos ********************")
print("-- Tiempo de entrenamiento: " + str(tr_time) + " s.")
print("-- Tiempo medio de segmentación de una imagen: " + str(np.mean(times_seg)) + " s.")
print("-- Tiempo medio de análisis de una imagen segmentada: " + str(np.mean(times_proc)) + " s.")
