# coding=UTF-8
from pyrobot.brain import Brain

import math
import cv2
import numpy as np
from sklearn import discriminant_analysis as da
import time

from lib import (analysis, binarize_image, geometry, hu_moments, mahalanobis, tr_img)

class BrainTestNavigator(Brain):
  # Captura de vídeo
  cap = None
  # Segmentador de imágenes (QDA)
  qda = None
  # Clasificador de iconos (distancia de Mahalanobis)
  maha = None
  # Guardar vídeo
  SAVE = True
  out = None
  # Ültimo punto indicado por la flecha
  last_arrow_pt = None
  # Último punto de entrada de la línea
  last_in = None
  # Último punto de salida de la línea
  last_out = None
  # Máximo error posible
  MAX_ERROR = 0
  # Lista de símbolos a reconocer
  symbols = ["Cruz", "Escaleras", "Servicio", "Telefono"]

  def setup(self):
    self.cap = cv2.VideoCapture(0)

    # Entrenamiento del segmentador de imágenes
    data, labels = tr_img.get_tr_img()
    self.qda = da.QuadraticDiscriminantAnalysis().fit(data, labels)

    # Entrenamiento del clasificador de iconos
    data, labels = hu_moments.get_db()
    self.maha = mahalanobis.classifMahalanobis().fit(data, labels)

    # Guardar vídeo
    width = int( self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int( self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    if self.SAVE:
      fourcc = cv2.VideoWriter.fourcc('X','V','I','D')
      self.out = cv2.VideoWriter('resources/videos/{}.avi'.format(time.time()), fourcc, 30, (width,height), True)

    self.MAX_ERROR = width

  def step(self):
    ret, img = self.cap.read()
    
    if not ret:
      raise Exception("Cannot self.cap video.")

    im_draw = img[img.shape[0]/4:,:,:]
    im_np = cv2.cvtColor(im_draw, cv2.COLOR_BGR2RGB)

    # Normalización RGB
    im_np = np.rollaxis((np.rollaxis(im_np,2)+0.0)/np.sum(im_np,2),0,3)[:,:,:2]
    # Filtro gaussiano
    # im_np = cv2.GaussianBlur(im_np, (0,0), 1)

    im_np = np.reshape(im_np, (im_np.shape[0]*im_np.shape[1],im_np.shape[2]))
    im_np = np.nan_to_num(im_np)

    labels_seg = np.reshape(self.qda.predict(im_np), (im_draw.shape[0], im_draw.shape[1]))

    # Comprobación de cruce
    cross = analysis.esCruce(im_draw,labels_seg)
    
    # Estimación de la orientación de la flecha
    if cross:
        arrow_pt = analysis.get_pt_flecha(im_draw, labels_seg, self.last_arrow_pt)
        self.last_arrow_pt = arrow_pt
    else:
        arrow_pt = None
        self.last_arrow_pt = None
        # Reconocimiento de símbolos
        # Binarización de la imagen
        im_bin, cont = binarize_image.binarize(labels_seg)
        if cont is not None:
            # Cálculo de los momentos de Hu
            hu_moments = hu_moments.get_hu(im_bin)
            # Clasificación del símbolo
            symbol = self.symbols[ self.maha.predict(hu_moments) ]
            # print(symbol)

    # Estimación de los bordes de la línea
    borders = analysis.get_bordes(im_draw, labels_seg)
    
    # Estimación de la entrada de la línea
    border_in = analysis.get_entrada(im_draw, borders, self.last_in)
    if border_in != -1:
        # Visualizar los píxeles de entrada en verde
        [ cv2.circle(im_draw,tuple(pt),2,(0,255,0),1) for pt in borders[border_in] ]
        pt_in = borders[border_in][len(borders[border_in])/2]
        self.last_in = pt_in
    else:
        pt_in = self.last_in

    # Estimación de la salida de la línea
    border_out = analysis.get_salida(borders, border_in, arrow_pt, self.last_out)
    if border_out != -1:
        # Visualizar los píxeles de salida en rojo
        [ cv2.circle(im_draw,tuple(pt),2,(0,0,255),1) for pt in borders[border_out] ]
        pt_out = borders[border_out][len(borders[border_out])/2]
        self.last_out = pt_out
    else:
        pt_out = self.last_out

    if self.SAVE:
        self.out.write(img)

    error = self.MAX_ERROR/2 - pt_out[0] + 0.0
    TV = error / self.MAX_ERROR
    FV = max(0, 1-abs(TV*1.5))
    print(FV, TV)
    self.move(FV, TV)

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
