# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 22:31:45 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:18:47 2018

@author: Administrator
"""
import numpy as np
import cv2
#尺寸
ds_factor =1
#開啟攝像頭
#cap = cv2.VideoCapture(0)
cap = cv2.imread('../data/cruise3.jpg')
cv2.imshow('ori eyes',cap)
if cap is None:
    raise IOError("Cannot open the webcam!")
#載入配置檔案
right_cascade = cv2.CascadeClassifier("../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")
#left_cascade = cv2.CascadeClassifier("../data/haarcascades/haarcascade_lefteye_2splits.xml")
#if left_cascade.empty():
 #   raise IOError('Unable to load the nose cascade classifier xml file')

while True:
    frame = cap
    frame = cv2.resize(frame,None,fx = ds_factor,
                       fy = ds_factor,
                       interpolation = cv2.INTER_CUBIC)    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #雙眼
    eyes_rects = right_cascade.detectMultiScale(gray,1.2,2,cv2.CASCADE_SCALE_IMAGE,(1,1))
    #nose_rects2 = left_cascade.detectMultiScale(gray)
    for (x,y,w,h) in eyes_rects:
        #cv2.rectangle(frame,(x,y),(w+x,h+y),(0,255,0),1)     
       # im2 = ImageGrab.grab(bbox = (int((x+w+x)/2)-30,int((y+y+h)/2)-30,int((x+w+x)/2)+30,int((y+y+h)/2)+30))
        bbox = (x,y,w,h)
        img = frame
        cv2.circle(frame,(int((x+w+x)/2),int((y+y+h)/2)),(int((w+h)*0.05)),(10,10,255),-1)
        #cv2.circle(frame,(int((x+w+x)/2),int((y+y+h)/2)),(5),(10,10,10),-1)
        #break
       
        mask = np.zeros(img.shape[:2],np.uint8)
        
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        rect = bbox
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]

    

    cv2.imshow('eyes',frame)
    #cv2.imwrite('CruiseEyes.jpg',img)
    cv2.imwrite('eyesRed.jpg',frame)
    
    if cv2.waitKey(1) == 27:
        break
    
#cap.release()
cv2.destroyAllWindows()