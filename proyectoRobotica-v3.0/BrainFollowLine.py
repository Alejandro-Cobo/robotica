# coding=UTF-8
from pyrobot.brain import Brain

import math
import cv2
import numpy as np
from sklearn import discriminant_analysis as da
import time

import lib

class BrainTestNavigator(Brain):
  # Video capture
  cap = None
  # Pixel segmentator (QDA)
  qda = None
  # Symbol classifier (Mahalanobis distance-based classifier)
  maha = None
  # Save video
  SAVE = False
  out = None
  # Frame counter
  im_count = 0
  # Last exit pointed by an arrow in a crossroad
  last_arrow = None
  # Last frame's end of the line
  last_in = None
  # Last frame's start of the line
  last_out = None


  def setup(self):
    self.cap = cv2.VideoCapture(0)

    # Train image segmentator
    data, labels = lib.classif.get_tr_img()
    self.qda = da.QuadraticDiscriminantAnalysis().fit(data, labels)

    # Train symbol classifier
    data, labels = lib.hu_moments.get_db()
    self.maha = lib.mahalanobis.classifMahalanobis().fit(data, labels)

    # Save video
    if SAVE:
      width = int( cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH) )
      height = int( cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) )
      fourcc = cv2.cv.CV_FOURCC(*'XVID')
      self.out = cv2.VideoWriter('resources/videos/{}}.avi'.format(time.time()), fourcc, 30, (width,height), True)

  def step(self):
    ret, img = capture.read()
    
    if not ret:
      raise Exception("Cannot capture video.")

    imDraw = img[img.shape[0]/4:,:,:]
    imNp = cv2.cvtColor(imDraw, cv2.COLOR_BGR2RGB)

    # Compute rgb normalization
    imNp = np.rollaxis((np.rollaxis(imNp,2)+0.0)/np.sum(imNp,2),0,3)[:,:,:2]
    # Apply a gaussian filter
    imNp = cv2.GaussianBlur(imNp, (0,0), 1)

    im2D = np.reshape(imNp, (imNp.shape[0]*imNp.shape[1],imNp.shape[2]))
    im2D = np.nan_to_num(im2D)

    labels_seg = np.reshape(seg.predict(im2D), (imDraw.shape[0], imDraw.shape[1]))

    if SAVE:
      paleta = np.array([[0,0,0],[0,0,255],[255,0,0]],dtype=np.uint8)
      im_seg = cv2.cvtColor(paleta[labels_seg],cv2.COLOR_RGB2BGR)
      self.out.write(im_seg)

    # Check if robot is in a crossroad
    cross = analisis.esCruce(imDraw,labels_seg)
    
    # Segment the arrow
    if cross:
        # cv2.putText(img, "Cruce detectado", (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        out_pt = analisis.get_pSalida(imDraw, labels_seg, last_arrow)
        last_arrow = out_pt
    else:
        # cv2.putText(img, "Sin cruces", (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        out_pt = None
        last_arrow = None

        # Reconocimiento de símbolos
        # Binarizo la imagen
        im_bin, cont = bin.binarize(labels_seg)
        if cont is not None:
            # Visualizar los contornos del símbolo
            # cv2.drawContours(imDraw, cont, -1, (255,0,0))
            # Calculo los momentos de Hu
            hu_moments = hu.get_hu(im_bin)
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
    # [ cv2.circle(imDraw,tuple(pt),2,(0,255,0),1) for pt in bordes[entrada] ]
    # Determino la salida de la línea
    salida = analisis.get_salida(bordes, entrada, out_pt, ultimaSalida)
    ultimaSalida = None
    if salida != -1:
        ultimaSalida = bordes[salida][len(bordes[salida])/2]
        # Visualizar los píxeles de salida en rojo
        # [ cv2.circle(imDraw,tuple(pt),2,(0,0,255),1) for pt in bordes[salida] ]
        pOut = bordes[salida][len(bordes[salida])/2]
        # Visualizar la línea que une la entrada y la salida
        # cv2.line(imDraw,tuple(pIn),tuple(pOut),(0,0,255,),2)
 
def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
	  engine.robot.requires("continuous-movement"))

  # If we are allowed (for example you can't in a simulation), enable
  # the motors.
  try:
    engine.robot.position[0]._dev.enable(1)
  except AttributeError:
    pass

  return BrainTestNavigator('BrainTestNavigator', engine)
