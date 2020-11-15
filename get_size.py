# get min max width height of images
# reference for preprocessing images 
import os
import cv2

lw=10000
uw=0
lh=10000
uh=0
for filename in os.listdir('./images'):
    try:
        im = cv2.imread('./images/'+str(filename))
        w, h, c = im.shape
        if(w<lw):
            lw=w
        if(w>uw):
            uw=w
        if(h>uh):
            uh=h
        if(h<lh):
            lh=h
    except:
    	print(filename)

print("minimum width size is "+str(lw)) #102
print("minimum height size is "+str(lh)) #243
print("maximum width size is "+str(uw)) #500
print("maximum height size is "+str(uh)) #500