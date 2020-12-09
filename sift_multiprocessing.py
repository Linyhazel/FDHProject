import cv2
from multiprocessing import Pool, Manager
from collections import defaultdict
import os, time, random
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def get_dist(filename, ides, matched_distance):
    # for every image, calculate its sift features
    img = cv2.imread('./data/images/'+str(filename),0) # flag=0, read as gray scale

        # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT (no mask)
    kp, des = sift.detectAndCompute(img,None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(ides, des)
    matches = sorted(matches, key = lambda x:x.distance)

    distance = 0
    for match in matches[:50]:
        distance += match.distance

    return filename, distance



if __name__=='__main__':
    input_img = input ("Enter path to image: ") 

    # compute sift for the input image
    iimg = cv2.imread(input_img,0) # flag=0, read as gray scale
    isift = cv2.xfeatures2d.SIFT_create()
    ikp, ides = isift.detectAndCompute(iimg,None)

    matched_distance = defaultdict(list)

    img_names = os.listdir('./data/images')

    print('Parent process %s.' % os.getpid())
    cpus = os.cpu_count() # 4 in my case
    pool = Pool(cpus)

    partial_func = partial(get_dist, ides=ides, matched_distance=matched_distance)
    result_map  = pool.map(partial_func, img_names)

    for filename, distance in (r for r in result_map if r is not None):
        matched_distance[filename].append(distance)

    #print('Waiting for all subprocesses done...')
    pool.close()
    pool.join()
    #print('All subprocesses done.')

    # find best match based on matching distance
    best_matched = min(matched_distance, key=matched_distance.get)
    print(best_matched)
    print(len(matched_distance))

    # show the mached image
    matched_img = cv2.imread('./data/images/'+best_matched,0)
    plt.imshow(matched_img),plt.show()