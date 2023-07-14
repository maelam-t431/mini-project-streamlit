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
