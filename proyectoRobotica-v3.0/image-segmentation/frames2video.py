import cv2
import numpy as np

cap = cv2.VideoCapture("frames/frame%02d.jpg")

while True:
    ret, frame = cap.read()
    if not ret or cv2.waitKey(100) == ord('q'):
        break
    cv2.imshow('Segmented video',frame)
    

cap.release()
cv2.destroyAllWindows()
