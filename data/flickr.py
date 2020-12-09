# get data from flickr
import flickrapi
# consider the region of lat: 45.395688-45.468072
#                        lon: 12.248631-12.396624

#input API key and secret
flickr=flickrapi.FlickrAPI('c5af07877cc00731fbec3b2fc063a9e4','4d9ffde5a58411d6',cache=True)

try:
    # for training data set
    photos = flickr.walk(text='Venice, building',has_geo=1,geo_context=2,extras='geo, url_m')

    # for testing data set
    #photos = flickr.walk(text='Venice, building',has_geo=0,geo_context=2,extras='geo, url_m')
except Exception as e:
    print('Error')

count = 0

file = open("venice_m1.txt","w+") 
#file = open("venice_test.txt","w+") 
for photo in photos:
    count += 1
    lat=photo.get('latitude')
    lon=photo.get('longitude')
    url=photo.get('url_m')

    if(str(lat) != None and str(lon) != None and str(url) != None):
        if(float(lat)>=45.395688 and float(lat)<=45.468072 and float(lon)>=12.248631 and float(lon)<=12.396624):
            line = str(lat)+","+str(lon)+","+str(url)
            file.write(line+"\n")
    # for test images
    #file.write(str(url)+"\n")

print(count)
file.close()