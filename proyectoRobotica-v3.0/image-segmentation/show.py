import cv2   

cam = cv2.VideoCapture(1)

while True:
    ret, frame = cam.read()

    if not ret:
        break
   
    cv2.imshow('Camera',frame)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()
