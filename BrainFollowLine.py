# coding=UTF-8
from pyrobot.brain import Brain

import math
import cv2
import numpy as np
from sklearn import discriminant_analysis as da
import time

import lib

class BrainTestNavigator(Brain):
  # Captura de vídeo
  cap = None
  # Segmentador de imágenes (QDA)
  qda = None
  # Clasificador de iconos (distancia de Mahalanobis)
  maha = None
  # Guardar vídeo
  SAVE = False
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
    data, labels = lib.tr_img.get_tr_img()
    self.qda = da.QuadraticDiscriminantAnalysis().fit(data, labels)

    # Entrenamiento del clasificador de iconos
    data, labels = lib.hu_moments.get_db()
    self.maha = lib.mahalanobis.classifMahalanobis().fit(data, labels)

    # Guardar vídeo
    width = int( cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH) )
    height = int( cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) )
    if self.SAVE:
      fourcc = cv2.cv.CV_FOURCC(*'XVID')
      self.out = cv2.VideoWriter('resources/videos/{}}.avi'.format(time.time()), fourcc, 30, (width,height), True)

    self.MAX_ERROR = width

  def step(self):
    ret, img = capture.read()
    
    if not ret:
      raise Exception("Cannot capture video.")

    im_draw = img[img.shape[0]/4:,:,:]
    im_np = cv2.cvtColor(im_draw, cv2.COLOR_BGR2RGB)

    # Normalización RGB
    im_np = np.rollaxis((np.rollaxis(im_np,2)+0.0)/np.sum(im_np,2),0,3)[:,:,:2]
    # Filtro gaussiano
    # im_np = cv2.GaussianBlur(im_np, (0,0), 1)

    im_np = np.reshape(im_np, (im_np.shape[0]*im_np.shape[1],im_np.shape[2]))
    im_np = np.nan_to_num(im_np)

    labels_seg = np.reshape(self.qda.predict(im_np), (im_draw.shape[0], im_draw.shape[1]))

    if self.SAVE:
      palette = np.array([[0,0,0],[0,0,255],[255,0,0]],dtype=np.uint8)
      im_seg = cv2.cvtColor(palette[labels_seg],cv2.COLOR_RGB2BGR)
      self.out.write(im_seg)

    # Comprobación de cruce
    cross = lib.analysis.esCruce(im_draw,labels_seg)
    
    # Estimación de la orientación de la flecha
    if cross:
        arrow_pt = lib.analysis.get_pSalida(im_draw, labels_seg, last_arrow_pt)
        self.last_arrow_pt = arrow_pt
    else:
        arrow_pt = None
        self.last_arrow_pt = None

        # Reconocimiento de símbolos
        # Binarización de la imagen
        im_bin, cont = bin.binarize(labels_seg)
        if cont is not None:
            # Cálculo de los momentos de Hu
            hu_moments = lib.hu.get_hu(im_bin)
            # Clasificación del símbolo
            symbol = self.symbols[ int(self.maha.predict(hu_moments)) ]
            # print(symbol)

    # Estimación de los bordes de la línea
    borders = lib.analysis.get_bordes(im_draw, labels_seg)

    # Estimación de la entrada de la línea
    border_in = lib.analysis.get_entrada(im_draw, borders, self.last_in)
    pt_in = borders[border_in][len(borders[border_in])/2]
    self.last_in = pt_in
    # Estimación de la salida de la línea
    border_out = lib.analysis.get_salida(borders, border_in, arrow_pt, self.last_out)
    if border_out != -1:
        pt_out = borders[border_out][len(borders[border_out])/2]
        self.last_out = p_out
    else:
      self.last_out = None

    # TODO: consigna de control teniendo en cuenta pt_in y pt_out.
    error = pt_out[0] - pt_in[0]
    TV = error / self.MAX_ERROR
    FV = max(0, 1-math.abs(TV*1.5))
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
