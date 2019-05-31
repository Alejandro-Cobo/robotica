# coding=UTF-8
from pyrobot.brain import Brain

import math
import cv2
import numpy as np
from sklearn import discriminant_analysis as da
import time

import os, sys
sys.path.append(os.getcwd())
from lib import (analysis, binarize_image, geometry, hu_moments, mahalanobis, tr_img)

class BrainTestNavigator(Brain):
  # Flag que indica si ha llegado al final del circuito
  stop = False
  # Flag que indica si debe esquivar un obstáculo
  avoid = False
  # Flag que indica si debe buscar la línea al principio del circuito
  search_line = True

  # Captura de vídeo
  cap = None
  # Segmentador de imágenes (QDA)
  qda = None
  # Clasificador de iconos (distancia de Mahalanobis)
  maha = None
  # Guardar vídeo
  SAVE_VIDEO = False
  out = None
  # Ültimo punto indicado por la flecha
  last_arrow_pt = None
  # Último punto de entrada de la línea
  last_in = None
  # Último punto de salida de la línea
  last_out = None
  # Máximo error posible
  MAX_ERROR = 0.0
  # Lista de símbolos a reconocer
  symbols = ["Cruz", "Escaleras", "Servicio", "Telefono"]

  # Máxima velocidad de avance
  MAX_SPEED = 0.7

  def setup(self):
    self.cap = cv2.VideoCapture(0)
    assert self.cap.isOpened()

    # Ajustar la resolución del vídeo a 320 x 180
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

    # Entrenamiento del segmentador de imágenes
    data, labels = tr_img.get_tr_data()
    self.qda = da.QuadraticDiscriminantAnalysis().fit(data, labels)

    # Entrenamiento del clasificador de iconos
    data, labels = hu_moments.get_db()
    self.maha = mahalanobis.classifMahalanobis().fit(data, labels)

    # Guardar vídeo
    width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if self.SAVE_VIDEO:
      fourcc = cv2.VideoWriter.fourcc('X','V','I','D')
      self.out = cv2.VideoWriter('resources/videos/{}.avi'.format(time.time()), fourcc, 5, (int(width),int(height)), True)
      self.out_seg = cv2.VideoWriter('resources/videos/{}_seg.avi'.format(time.time()), fourcc, 5, (int(width),int(height)), True)

    self.MAX_ERROR = width / 2

  def step(self):
    # Quedarse quieto si ha terminado el circuito
    if self.stop:
      return

    ############# Segmentación y análisis de imágenes #############
    ret, img = self.cap.read()
    
    assert ret, "Cannot read from video capture."

    im_draw = img
    # im_draw = img[img.shape[0]/4:,:,:]
    im_np = cv2.cvtColor(im_draw, cv2.COLOR_BGR2RGB)

    # Normalización RGB
    im_np = np.rollaxis((np.rollaxis(im_np,2)+0.0)/np.sum(im_np,2),0,3)[:,:,:2]
    
    # Filtro gaussiano
    im_np = cv2.GaussianBlur(im_np, (0,0), 1)

    im_np = np.reshape(im_np, (im_np.shape[0]*im_np.shape[1],im_np.shape[2]))
    im_np = np.nan_to_num(im_np)

    labels_seg = np.reshape(self.qda.predict(im_np), (im_draw.shape[0], im_draw.shape[1]))

    # Comprobación de cruce
    cross = analysis.es_cruce(im_draw,labels_seg)

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
          hu = hu_moments.get_hu(im_bin)
          # Clasificación del símbolo
          symbol = self.symbols[ self.maha.predict(hu) ]
          # Visualizar el símbolo reconocido
          # cv2.putText(img, symbol,(10,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
          print(symbol)
          if symbol == "Cruz":
            self.move(0, 0)
            self.stop = True
            return

    # Estimación de los bordes de la línea
    borders = analysis.get_bordes(im_draw, labels_seg)
    
    # Estimación de la entrada de la línea
    border_in = analysis.get_entrada(im_draw, borders, self.last_in)
    if border_in != -1:
        # Visualizar los píxeles de entrada en verde
        # [ cv2.circle(im_draw,tuple(pt),2,(0,255,0),1) for pt in borders[border_in] ]
        pt_in = borders[border_in][len(borders[border_in])/2]
        self.last_in = pt_in
    else:
        pt_in = self.last_in

    # Estimación de la salida de la línea
    border_out = analysis.get_salida(borders, border_in, arrow_pt, self.last_out)
    if border_out != -1:
        # Visualizar los píxeles de salida en rojo
        # [ cv2.circle(im_draw,tuple(pt),2,(0,0,255),1) for pt in borders[border_out] ]
        pt_out = borders[border_out][len(borders[border_out])/2]
        self.last_out = pt_out
    else:
        pt_out = self.last_out

    hasLine = pt_out is not None

    # Guardar vídeo
    if self.SAVE_VIDEO:
        paleta = np.array([[0,0,0],[0,0,255],[255,0,0]],dtype=np.uint8)
        imSeg = cv2.cvtColor(paleta[labels_seg],cv2.COLOR_RGB2BGR)
        self.out.write(img)
        self.out_seg.write(imSeg)

    ############# Consignas de control #############
    # Buscar la línea
    if not hasLine and self.search_line:
      self.move(self.MAX_SPEED, 0)
      return

    elif self.search_line:
      self.search_line = False
    
    # Esquivar obstáculos
    front = min([s.distance() for s in self.robot.range["front"]])   
    if front < 0.2:
       self.avoid = True
       self.move(0, 1)
       return

    if self.avoid:
      if hasLine:
        self.avoid = False
        return
      dist = min([s.distance() for s in self.robot.range["front-right"]])
      if dist == 0:
        self.move(-self.MAX_SPEED, 0)
        return
      error = 0.1 - dist
      if error < 0:
        error /= 3
      else:
        error /= 0.1
      TV = max(-1, error)
      FV = min(self.MAX_SPEED, max(0, 1-abs(TV)))
      self.move(FV, TV)
      return
    
    # Seguir la línea
    error = self.MAX_ERROR - pt_out[0]
    TV = error / self.MAX_ERROR
    FV = min(self.MAX_SPEED, max(0, 1-abs(TV)))
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
