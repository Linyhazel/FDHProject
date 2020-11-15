import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_img = input ("Enter path to image: ") 


# compute sift for the input image
iimg = cv2.imread(input_img,0) # flag=0, read as gray scale
isift = cv2.xfeatures2d.SIFT_create()
ikp, ides = isift.detectAndCompute(iimg,None)


kp_dic = {}
des_dic = {}
match_dic = {}
matched_distance = {}

# for every image, calculate its sift features
for filename in os.listdir('./images'):
    img = cv2.imread('./images/'+str(filename),0) # flag=0, read as gray scale

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT (no mask)
    kp, des = sift.detectAndCompute(img,None)

    kp_dic[filename] = kp
    des_dic[filename] = des

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(ides, des)
    matches = sorted(matches, key = lambda x:x.distance)
    match_dic[filename] = matches[:50]

    distance = 0
    for match in matches[:50]:
        distance += match.distance

    matched_distance[filename] = distance

# find best match based on matching distance
best_matched = min(matched_distance, key=matched_distance.get)
print(best_matched)

# show the mached image and the matches
matched_img = cv2.imread('./images/'+best_matched,0)
result = cv2.drawMatches(iimg, ikp, matched_img, kp_dic[best_matched], match_dic[best_matched], matched_img, flags=2)
plt.imshow(result),plt.show()