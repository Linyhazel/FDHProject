import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras, cv2

# load model
@st.cache()
def load_model():
    model = keras.models.load_model("model/model.h5")

    maxlatlon = np.array([45.46308, 12.377772])
    minlatlon = np.array([45.404235, 12.258982])
    return model, maxlatlon, minlatlon

# predict the location of an image
def predict(img):
    model, maxlatlon, minlatlon = load_model()
    
    # preprocess the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # get the predicted position
    predict = model.predict((img.astype(np.float32) / 255).reshape(1, 224, 224, 3))
    predict = predict * (maxlatlon - minlatlon) + minlatlon

    # preprocess the coordinate data
    map_data = pd.DataFrame(
       predict,
       columns = ['lat', 'lon'])

    # show the position in the map
    st.map(map_data)
    return

# body
st.title("Paintings and Photos Geolocalisation")

st.markdown("A method for recognising the places images represent and repositioning them on the map.")

st.markdown("*photo 1 -> photo 3 from left to right*")

# divided into three columns
img_1, img_2, img_3 = st.beta_columns(3)
img_1.image("images/test1.jpg", use_column_width = True)
img_2.image("images/test2.jpg", use_column_width = True)
img_3.image("images/test3.jpg", use_column_width = True)

st.markdown("*painting 1 -> painting 3 from left to right*")

img_4, img_5, img_6 = st.beta_columns(3)
img_4.image("images/test4.jpg", use_column_width = True)
img_5.image("images/test5.jpg", use_column_width = True)
img_6.image("images/test6.jpg", use_column_width = True)

st.subheader("**The Location of the Picture**")

# sidebar
st.sidebar.title("Find Where Are These Scenery")

img_select = st.sidebar.selectbox(
    'Get the location of the scenery in these pictures now!',
     ['Venice photo 1', 'Venice photo 2', 'Venice photo 3',
      'Venice painting 1', 'Venice painting 2', 'Venice painting 3'])

st.sidebar.title("Upload Your Venice Photo")

uploaded_file = st.sidebar.file_uploader("Choose an image file. NOTE: An abstractionism work is not recommended.", type = "jpg")

# show and predict the uploaded image
if uploaded_file is not None:
    img_up = plt.imread(uploaded_file)
    st.sidebar.image(img_up, use_column_width = True)
    predict(img_up)
# predict the selected image
else:
    if img_select == "Venice photo 1":
        img_demo = plt.imread("images/test1.jpg")
        predict(img_demo)
    elif img_select == 'Venice photo 2':
        img_demo = plt.imread("images/test2.jpg")
        predict(img_demo)
    elif img_select == 'Venice photo 3':
        img_demo = plt.imread("images/test3.jpg")
        predict(img_demo)
    elif img_select == 'Venice painting 1':
        img_demo = plt.imread("images/test4.jpg")
        predict(img_demo)
    elif img_select == 'Venice painting 2':
        img_demo = plt.imread("images/test5.jpg")
        predict(img_demo)
    elif img_select == 'Venice painting 3':
        img_demo = plt.imread("images/test6.jpg")
        predict(img_demo)
    else:
        st.error("error!")

st.subheader("**Useful Links**")
st.markdown("Find our wikipedia here:")
st.markdown("http://fdh.epfl.ch/index.php/Paintings_/_Photos_geolocalisation")

st.markdown("Find our code here:")
st.markdown("https://github.com/Linyhazel/FDHProject")

st.markdown("")
st.image('images/poem.jpg', use_column_width = True)

st.markdown("")
st.text('This project was made as part of the course of "Foundation of Digital Humanities" in EPFL.')
st.text("The team: Yuanhui Lin, Zijun Cui, Junhong Li")