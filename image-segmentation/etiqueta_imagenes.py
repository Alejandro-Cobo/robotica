import cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import select_pixels as sel
import sys

# Abres el video / camara con
video = sys.argv[1]
if video == None:
    capture = cv2.VideoCapture()
else:
    capture = cv2.VideoCapture(video)

# Lees las imagenes y las muestras para elegir la(s) de entrenamiento
# posibles funciones a usar

key = 0

while key != ord('q'):
    ret, imNp = capture.read()
    cv2.imshow('Captura',imNp)
    key = cv2.waitKey(150)

capture.release()
cv2.destroyWindow("Captura")

# Si deseas mostrar la imagen con funciones de matplotlib posiblemente haya que cambiar
# el formato, con
imNp = cv2.cvtColor(imNp, cv2.COLOR_BGR2RGB)

# Esta funcion del paquete "select_pixels" pinta los pixeles en la imagen 
# Puede ser util para el entrenamiento

markImg = sel.select_fg_bg(imNp)

# Tambien puedes mostrar imagenes con las funciones de matplotlib
plt.imshow(markImg)
plt.show()

# Si deseas guardar alguna imagen ....
imsave('resources/imgs/img.png',imNp)
imsave('resources/imgs/img-paint.png',markImg)
