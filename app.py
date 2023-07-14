import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np


# Define the model architecture
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Load the model weights from file
model.load_weights('model.h5')


st.set_page_config(layout="wide", page_title="MNIST Classifier")

st.write("## Classify your images")
st.write(
    "Try uploading an image of a handwritten digit, the model will predict which digit it is."
)
st.sidebar.write("## Upload your image")


def fix_image(img):
    image = Image.open(img)
    img = image.convert('L')  # Convert image to grayscale
    #resize the image to 28x28
    img = img.resize((28, 28))
    #convert the image to a numpy array
    x = np.array(img)
    #expand the dimensions of the array to match the input shape of the model
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=-1)
    #normalize the image
    x = x / 255.0

    # Perform inference
    