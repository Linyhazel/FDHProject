# download images given urls
import os
import requests

def download(url, name):
    r = requests.get(url)
    path = "./images/"
    with open(path+name, "wb") as f:
        f.write(r.content)


outfile=open("label.txt","w")
f = open("m1_dd.txt","r")
file_name = 0
for line in f:
    ele = line.split(',')
    download(ele[2], str(file_name)+".jpg")
    outfile.write(ele[0]+","+ele[1]+","+str(file_name)+".jpg\n")
    file_name+=1
    
outfile.close()