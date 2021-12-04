import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

with_mask = with_mask.reshape(200, 50 * 50 * 3)
without_mask = without_mask.reshape(200, 50 * 50 * 3)

X = np.r_[with_mask, without_mask]

labels = np.zeros(X.shape[0])
labels[200:] = 1.0
names = {0: 'Mask', 1.0: 'No Mask'}

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.20)

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


svm = SVC()
svm.fit(x_train, y_train)

haar_data = cv2.CascadeClassifier('haar_data.xml')

# y_pred = svm.predict(x_test)
# print(accuracy_score(y_test, y_pred))

capture = cv2.VideoCapture(0)  # initialize camera
data = []  # face data
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, image = capture.read()  # read video frame
    if flag:  # check if camera is available
        faces = haar_data.detectMultiScale(image)  # detecting faces
        for x, y, w, h in faces:  # looping through faces
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255),
                          4)  # drawing reactange on face
            face = image[y: y + h, x: x + w, :]  # slicing faces
            face = cv2.resize(face, (50, 50))  # resizing faces to 50 x 50
            face = face.reshape(1, -1)
            face = pca.transform(face)
            pred = svm.predict(face)[0]
            n = names[int(pred)]
            cv2.putText(image, n, (x, y + h + 30), font, 1, (244, 250, 250), 2)
        cv2.imshow('Face Mask Detector', image)
        if (cv2.waitKey(2) == 27):
            break

capture.release()  # release camera used by openCv
cv2.destroyAllWindows()  # close all windows opened by openCv
