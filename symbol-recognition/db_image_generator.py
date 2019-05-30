# coding=UTF-8
import cv2
from scipy.misc import imread, imsave
import numpy as np
from sklearn import discriminant_analysis as da
import os, os.path

from lib import binarize_image as bin
from lib import tr_img

IMG_DB = os.path.abspath("./resources/dataset") + "/"
VID_DB = os.path.abspath("./resources/videos") + "/"

def rename_files(path):
    files = [name for name in os.listdir(path) ]
    for index, file in enumerate(files):
        os.rename(os.path.join(path, file), os.path.join(path, "%03d.jpg" % (index+1)))
    
    files = [name for name in os.listdir(path) ]
    for index, file in enumerate(files):
        os.rename(os.path.join(path, file), os.path.join(path, "img%03d.jpg" % (index+1)))

def add_db_image(img, folder):
    save_im_count = len([name for name in os.listdir(IMG_DB + folder)]) + 1
    cv2.imwrite(IMG_DB + folder + "/img%03d.jpg" % save_im_count, img)
    print("Saved image: " + IMG_DB + folder + "/img%03d.jpg" % save_im_count)

data, labels = tr_img.get_tr_data()
seg = da.QuadraticDiscriminantAnalysis().fit(data, labels)

folders = ["cruz","escaleras","servicio","telefono"]
print("Folders:")
for i in range(len(folders)):
    print("\t[" + str(i) + "] " + folders[i])
folder = folders[ input("* Select a folder [0,1,2,3]: ") ]
videos = [name for name in os.listdir(VID_DB + folder) ]

rename_files(IMG_DB + folder)
save_im_count = len([name for name in os.listdir(IMG_DB + folder)]) + 1
print("Current number of files in " + IMG_DB + folder + ": " + str(save_im_count-1))

print("Videos in " + VID_DB + folder + ":")
for i in range(len(videos)):
    print("\t[" + str(i) + "] " + videos[i])
video = videos[ input("* Select a video: ") ]

capture = cv2.VideoCapture(VID_DB + folder + "/" + video)

im_count = 0
while True:
    ret, img = capture.read()
    im_count += 1

    if not ret:
        break

    try:
        imNp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imNp = np.rollaxis((np.rollaxis(imNp,2)+0.0)/np.sum(imNp,2),0,3)[:,:,:2]
        # imNp = cv2.GaussianBlur(imNp, (0,0), 1)
        im2D = np.reshape(imNp, (imNp.shape[0]*imNp.shape[1],imNp.shape[2]))
        labels_seg = np.reshape(seg.predict(im2D), (img.shape[0], img.shape[1]))
    except:
        continue
    
    img, _ = bin.binarize(labels_seg)
    if img is None:
        continue
    
    cv2.imshow("Video", img)
    
    k = cv2.waitKey(1)
    if k == ord(' '):
        add_db_image(img, folder)
    if k == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
