import tensorflow.keras
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

def getData(labelpath, picpath):
	labels = pd.read_csv(labelpath, sep=",", header=None)
	labels.columns = ["lat", "lon", "pic"]
	outputs = labels[["lat", "lon"]]
	pic_files = labels["pic"]

	inputs = []

	for file in pic_files:
		# load the image, and resize it to be a fixed
		# 224x224 pixels while ignoring aspect ratio
		image = cv2.imread(picpath+file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (224, 224)).astype(np.float32)
		# update the input lists
		inputs.append(image)


	# convert the inputs and outputs to NumPy arrays
	inputs = np.array(inputs)
	outputs = np.array(outputs)

	return [inputs, outputs]


# read data for training
[data, labels] = getData("label.txt","./images/")

# normalize data
data = data/255

# normalize labels using min-max
maxlatlon = np.max(labels,axis=0)
minlatlon = np.min(labels,axis=0)
normalized_labels = (labels-minlatlon)/(maxlatlon-minlatlon)

(trainX, testX, trainY, testY) = train_test_split(data, normalized_labels, test_size=0.1, random_state=42)


def custom_loss_function(y_true, y_pred):
   difference = tf.abs(y_true - y_pred)
   return tf.reduce_sum(difference, axis=-1)

# construct modified resnet
baseModel = keras.applications.ResNet101(weights="imagenet", include_top=False,	input_tensor=Input(shape=(224, 224, 3)))

# get the output before the classifier
base_output = baseModel.output

# change classifier to regressor output dimension is 2
base_output = AveragePooling2D(pool_size=(4, 4))(base_output)
base_output = Flatten(name="flatten")(base_output)
base_output = Dropout(0.2)(base_output)
base_output = Dense(2, activation='linear', kernel_initializer='normal')(base_output)

model = Model(inputs=baseModel.input, outputs=base_output)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

model.compile(loss="mse", optimizer="adam")

print("[INFO] training head...")
history = model.fit(trainX, trainY, epochs=35, batch_size=128, shuffle=True, validation_data=(testX, testY))


import matplotlib.pyplot as plt

image = cv2.imread("test1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
test = cv2.resize(image, (224, 224))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), dpi=100, sharex=True, sharey=True)
ax.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
ax.axis('off')

predictY = model.predict(test.reshape(1,224,224,3))

predictY = predictY*(maxlatlon-minlatlon)+minlatlon
print(predictY)