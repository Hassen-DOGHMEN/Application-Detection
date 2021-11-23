import cv2
import numpy as np
import time
import random
from os.path import dirname, join
output_file_cropped = r"C:\Users\21658\PycharmProjects\notion_face_detection\dnn_fullHD_Facedetection\\"

modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"
configfile = join(dirname(__file__), configFile)

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


fileFolder = "../examples/faces/"

video_capture = cv2.VideoCapture(0)
nbre_frame = 0
temps_total = 0
while True:
    ret, frame = video_capture.read()
    nbre_frame = nbre_frame + 1
    img = cv2.resize(frame, (300, 300))

    h, w = img.shape[:2]
    start = time.time()
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
    (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    laps = time.time() - start
    print ("---frame number---",nbre_frame,"---execution time ---",laps)
    temps_total = temps_total + laps
    #to draw faces on image
    for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                l = random.randint(10000000, 99999999)
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
                crop_img = img[y:y1,x:x1]
                cv2.imshow('croped image',crop_img)
    time.sleep(1)
    #cv2.imwrite(output_file_cropped + str(l) + '.jpg', crop_img)
    cv2.imshow('my webcam', img)
    cv2.waitKey(10)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    #input("Press Enter to continue...")
video_capture.release()
cv2.destroyAllWindows()
#print ("-----temps moyenne-----",temps_total/nbre_frame)