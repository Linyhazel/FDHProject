# FDHProject - Paintings and Photos Geolocalisation

## Introduction
A method for recognising the places images represent and repositioning them on the map.

## Dataset
We used the python package flickrapi to crawl about 2390 photos with geo-coordinates inside Venice from Flickr. You can find a part of these photos in 'data/images/' and the complete 'label.txt' which contains all the coordinates corresponding to the photos in the training set. Since the 'images' directory was truncated to 1000 files, you can download the complete training set here:

https://drive.google.com/drive/folders/13jphiHfbIzM11ZPx0ShYPu-NKwCkHE6P?usp=sharing

However, you do not need to download the dataset if you just want to see the final effect, because we provide you with the trained model. You can find the model link below.

## Requirements
This project is based on python3.7 and above.

To run the web_app, you need to import [Streamlit](https://www.streamlit.io/) library: 

`<$ pip install streamlit>`

Then download the web_app file and run the .py file

`<streamlit run web_app.py>`

## Trained model
Due to the upload size limitation, we did not upload our trained model in this repository. You can download the trained model here:

https://drive.google.com/file/d/1wDbxPs0jIri-ea6WdRDsP2F88iPDfQ2s/view?usp=sharing

Then add it to the ‘web app/model/’ path.

To train a new model, download the dataset(photos and labels) and run resnet.ipynb.

## Wikipedia
Find the wiki of this project here:

http://fdh.epfl.ch/index.php/Paintings_/_Photos_geolocalisation

## Contributors
Junhong Li, Yuanhui Lin, Zijun Cui
