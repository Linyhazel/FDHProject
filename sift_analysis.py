# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:55:19 2020

@author: JUNHONG
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from matplotlib import pyplot as plt 
import seaborn as sns

def bootstrap_CI(data, nbr_draws):
    median = np.zeros(nbr_draws)
    data = np.array(data)

    for n in range(nbr_draws):
        indices = np.random.randint(0, len(data), len(data))
        data_tmp = data[indices] 
        median[n] = np.nanmedian(data_tmp)

    return [np.nanpercentile(median, 2.5),np.nanpercentile(median, 97.5)]

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

fs = './data/label.txt'
label = Readlabel(fs)

test = pd.read_table('./sift assessment results/test.txt',header = None,sep = '\t')
test = test.values.tolist()
match = pd.read_table('./sift assessment results/match.txt',header = None,sep = '\t')
match = match.values.tolist()

error = []

for index, item in enumerate(test):
    error.append(img2error(item[0],match[index][0],label))
    
bootstrap_CI(error, 10000)

plt.title('distribution of error')
plt.xlabel('error')
sns.displot(error, kde=False)
plt.show()