# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:11:39 2020

@author: JUNHONG
"""

import pandas as pd

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
    
    print(Eu_error(float(lat1),float(lon1),float(lat2),float(lon2)))

def main():
    fs = 'label2.txt'
    label = Readlabel(fs)
    img1 = '0.jpg'
    img2 = '1.jpg'
    print(img2error(img1,img2,label))
    

main()