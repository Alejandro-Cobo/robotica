import cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

ret, img = capture.read()
cv2.imshow("Imagen", img)
cv2.waitKey(1)
