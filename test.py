import os
import cv2
import numpy as np
import json

img = cv2.imread('./images/0.jpg',0) # flag=0, read as gray scale
img2 = cv2.imread('./images/2.jpg',0) # flag=0, read as gray scale

sift = cv2.xfeatures2d.SIFT_create()

kp, des = sift.detectAndCompute(img,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(des,des2)
matches = sorted(matches, key = lambda x:x.distance)

distance = 0
for match in matches[:50]:
	distance += match.distance

print(distance)