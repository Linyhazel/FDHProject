# draw distribution of the data 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

def drawMap():
    inFile = open("m1_dd.txt",'r')
    lon = []
    lat = []
    for line in inFile:
        el = line.split(',')
        lon.append(float(el[1]))
        lat.append(float(el[0]))
    
    fig = plt.gcf()
    
    map = Basemap(projection='lcc',  
                  lat_0=45.4336575,
                  lon_0=12.318351, 
                  urcrnrlat=45.46308, 
                  llcrnrlat=45.404235,  
                  urcrnrlon=12.377772,  
                  llcrnrlon=12.258982, 
                  resolution='c', 
                  area_thresh=10000, 
                  rsphere=6371200.)

    fig.set_size_inches(30,  30)

    parallels = np.arange(0., 90, 10.) 
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10) # 绘制纬线
    
    meridians = np.arange(80., 140., 10.)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10) # 绘制经线
    
    
    x, y = map(lon, lat)
    
    map.scatter(x, y, s=10, color='red')
    
    plt.title("flick point in Venice")
    
    plt.show()
    
    inFile.close()
    
# region is too small to draw map using provided api    
drawMap()