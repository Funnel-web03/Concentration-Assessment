import numpy as np
import cv2 as cv
import dlib 
import matplotlib
import matplotlib.pyplot as plt
import sys
from skimage import feature
import datetime 
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import os
import pickle
import imageio

#cv.namedWindow("Tracking")
'''vc = cv.VideoCapture(0)
if (vc.isOpened()):
    ret,frame = vc.read()
else:
    ret = False
while ret:
    cv.imshow("Tracking",frame)
    ret,frame = vc.read()
    k = cv.waitKey(20)
    if k == 27:
        break
cv.destroyAllWindows()'''

faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)
while True:
    ret,frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces :
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv.imshow("Concetration",frame)
    k = cv.waitKey(10)
    if k == 27 :
        break
cap.release()
cv.destroyAllWindows()

faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)
count=1
path='C:\\Project\\Project\\Model_train\\Focused'
while count<401:
    ret,frame = cap.read()
    count+=1
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces :
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        sub_face = frame[y:y+h,x:x+h]
        cv.imwrite(os.path.join(path,'ffaces'+str(count)+'.jpg'),sub_face)
    frame = cv.putText(frame,"Traing : Focused ",(30,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,155,140),2,cv.LINE_4)
    cv.imshow("Concetration",frame)
    k = cv.waitKey(10)
    if k == 27 :
        break
cap.release()
cv.destroyAllWindows()

cap = cv.VideoCapture(0)
count = 1
path='C:\\Project\\Project\\Model_train\\Not_Focused'
while count<351:
    ret,frame = cap.read()
    count+=1
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces :
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        sub_face = frame[y:y+h,x:x+h]
        cv.imwrite(os.path.join(path,'ffaces'+str(count)+'.jpg'),sub_face)
    frame = cv.putText(frame,"Traing : Not Focused ",(30,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,155,140),2,cv.LINE_4)
    cv.imshow("Concetration",frame)
    k = cv.waitKey(10)
    if k == 27 :
        break
cap.release()
cv.destroyAllWindows()

class LocalBinaryPatterns:
    def __init__(self,numPoints,radius):
        self.numPoints = numPoints
        self.radius = radius
    def describe(self,image,eps=1e-7):
        lbp = feature.local_binary_pattern(image,self.numPoints,self.radius,method="uniform")
        (hist,_)=np.histogram(lbp.ravel(),bins=np.arange(0,self.numPoints+3),range=(0,self.numPoints+2))
        hist = hist.astype("float")
        hist/=(hist.sum()+eps)
        return hist

p = 24
r = 8
desc = LocalBinaryPatterns(p,r)
data=[]
labels=[]

for cls in os.listdir("C:/Project/Project/Model_train"):
    clspath = os.path.join("C:/Project/Project/Model_train",cls)
    for file in os.listdir(clspath):
        imagepath = os.path.join(clspath,file)
        image = cv.imread(imagepath)
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        #extract label from image path and update label to label list
        labels.append(cls)
        data.append(hist)
 
#training Linear SVM
model = LinearSVC(C=10)
model.fit(data,labels)
pickle.dump(model,open("C:/Project/Project/train.pkl",'wb'))

#teasting SVM
cv.namedWindow("Preview")
model = pickle.load(open("C:/Project/Project/train.pkl",'rb'))
desc = LocalBinaryPatterns(24,8)
faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
vc1 = cv.VideoCapture(0)
timestamps=[]
status=[]
while True:
    ret,frame = vc1.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        if(w>120 and h>120):
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            sub_faces = frame[y:y+h,x:x+w]
            gray = cv.cvtColor(sub_faces,cv.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            prediction = model.predict(hist.reshape(1,-1))
            cv.putText(frame,prediction[0],(x,y),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv.LINE_4)
            timestamps.append(datetime.datetime.now().time)
            status.append(prediction[0])
    cv.imshow("preview",frame)
    k = cv.waitKey(20)
    if k == 27:
        break
count=1
vc1.release()
cv.destroyAllWindows()