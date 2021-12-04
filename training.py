import cv2
import numpy as np

haar_data = cv2.CascadeClassifier('haar_data.xml')

capture = cv2.VideoCapture(0)  # initialize camera
data = []  # face data
while True:
    flag, image = capture.read()  # read video frame
    if flag:  # check if camera is available
        faces = haar_data.detectMultiScale(image)  # detecting faces
        for x, y, w, h in faces:  # looping through faces
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255),
                          4)  # drawing reactange on face
            face = image[y: y + h, x: x + w, :]  # slicing faces
            face = cv2.resize(face, (50, 50))  # resizing faces to 50 x 50
            if len(data) < 200:
                data.append(face)
        cv2.imshow('Face Mask Detector', image)
        if (cv2.waitKey(2) == 27) or len(data) >= 200:
            break

capture.release()  # release camera used by openCv
cv2.destroyAllWindows()  # close all windows opened by openCv

np.save('with_mask.npy', data)
