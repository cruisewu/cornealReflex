
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 20:25:59 2018
@author: Miracle
"""
#检测瞳孔
import cv2
import math
#打开图片
img = cv2.imread('../data/cruise3.jpg')
#放缩尺寸
scaling_factor = 0.85
 
img = cv2.resize(img,None,
               fx = scaling_factor,
               fy = scaling_factor,
               interpolation = cv2.INTER_AREA)
 
cv2.imshow('Original Image',img)
gray = cv2.cvtColor(~img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray Image',gray)
 
rett,thresh_gray = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
cv2.imshow('thresh_gray Image',thresh_gray)
thresh_gray, contours, hierarchy = cv2.findContours(thresh_gray,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
cv2.imshow('thresh_gray image',thresh_gray)
 
for contour in contours:
    area = cv2.contourArea(contour)
    rect = cv2.boundingRect(contour)
    x,y,width,height = rect
    radius = 0.25*(width+height)
    
    area_condition = (100 <= area <= 200)
    symetry_condition = (abs(1-float(width)/float(height))<=0.2)
    fill_condition = (
            abs(1-(area/(math.pi*math.pow(radius,2.0))))<=0.3)
    
    if area_condition and symetry_condition and fill_condition:
        cv2.circle(img,
                   (int(x+radius),int(y+radius)),
                   int(1.3*radius),
                   (0,0,255),
                   -1)
cv2.imshow('Pupil',img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
