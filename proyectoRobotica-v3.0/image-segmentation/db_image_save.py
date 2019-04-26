import cv2
import numpy as np
import time    

folder1 = "cruz/"
folder2 = "escaleras/"
folder3 = "servicio/"
folder4 = "telefono/"

################################
folder = folder1
################################

filename = "video3.avi"

cam = cv2.VideoCapture(0)
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cam.get(cv2.CAP_PROP_FPS)
if fps < 1:
    print("Estimando fps...")
    nFrames = 120
    start = time.time()
    for i in xrange(0,nFrames):
        ret, frame = cam.read()
    seconds = time.time() - start
    fps = nFrames / seconds
print("{0} x {1} @ {2} fps".format(width, height, fps))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('dataset/'+folder+filename, fourcc, fps, (width,height), True)

record = False

while True:
    ret, frame = cam.read()

    if not ret:
        break

    img = np.copy(frame)
    if record:
        cv2.putText(img, "Grabando en " + folder + filename, (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
    cv2.imshow('Camera',img)
    k = cv2.waitKey(1)

    if k == ord(' '):
        record = not record

    elif k == ord('q'):
        break
        
    if record:
        out.write(frame)

out.release()
cam.release()
cv2.destroyAllWindows()
