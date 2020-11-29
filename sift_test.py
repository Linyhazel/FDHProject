import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def img2XY(img, dataframe):
    lat = dataframe[dataframe['img']==img]['lat']
    lon = dataframe[dataframe['img']==img]['lon']
    
    return lat,lon

def Eu_error(lat1,lon1,lat2,lon2):
    distance = (lat1-lat2)**2+(lon1-lon2)**2
    
    return distance

def Readlabel(fs):
    label = pd.read_table(fs,header = None,sep = '\t')
    label.columns = ['X']
    ppp = label['X'].str.split(',',expand = True)
    ppp.columns = ['lat','lon','img']
    

    #print(label.size)
    #print(ppp.head())
    #print(type(ppp['img'].head(0)))
    return ppp
    
def img2error(img1,img2,label):

    #img1 = '0.jpg'
    #img2 = '1.jpg'
    lat1, lon1 = img2XY(img1, label)
    lat2,lon2 = img2XY(img2,label)
    
    return Eu_error(float(lat1),float(lon1),float(lat2),float(lon2))
#input_img = input ("Enter path to image: ") 
time_start=time.time()

fs = 'label.txt'
label = Readlabel(fs)

imgs = label['img']
test_imgs = imgs.sample(int(0.02*len(imgs)))
test_imgs_index = test_imgs.index
all_index = imgs.index
match_imgs_index = all_index.difference(test_imgs_index)
match_imgs = imgs.loc[match_imgs_index]


# compute sift for the input image
# iimg = cv2.imread(input_img,0) # flag=0, read as gray scale
# isift = cv2.xfeatures2d.SIFT_create()
# ikp, ides = isift.detectAndCompute(iimg,None)




# for every image, calculate its sift features
#for filename in os.listdir('./images'):
test_list = []
match_list = []
for input_img in test_imgs:
    test_list.append(input_img)
    
    iimg = cv2.imread('./images/'+str(input_img),0) # flag=0, read as gray scale
    isift = cv2.xfeatures2d.SIFT_create()
    ikp, ides = isift.detectAndCompute(iimg,None)
    
    kp_dic = {}
    des_dic = {}
    match_dic = {}
    matched_distance = {}
    for filename in match_imgs:
        img = cv2.imread('./images/'+str(filename),0) # flag=0, read as gray scale

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
    
        # find the keypoints and descriptors with SIFT (no mask)
        kp, des = sift.detectAndCompute(img,None)
    
        kp_dic[filename] = kp
        des_dic[filename] = des
    
        #feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

#ides = []
#des = []

    
        matches = bf.match(ides, des)
        matches = sorted(matches, key = lambda x:x.distance)
        match_dic[filename] = matches[:50]

        distance = 0
        for match in matches[:50]:
            distance += match.distance

        matched_distance[filename] = distance

# find best match based on matching distance
    best_matched = min(matched_distance, key=matched_distance.get)
    #print(best_matched)
    match_list.append(best_matched)

time_end=time.time()
print('totally cost',time_end-time_start)
# show the mached image and the matches
#matched_img = cv2.imread('./images/'+best_matched,0)
#result = cv2.drawMatches(iimg, ikp, matched_img, kp_dic[best_matched], match_dic[best_matched], matched_img, flags=2)
#plt.imshow(result),plt.show()