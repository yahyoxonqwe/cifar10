import tensorflow as tf
from tensorflow import keras
import streamlit as st
import cv2
import numpy as np

st.cache_data()
with st.spinner('Model is being loaded..'):
    model = keras.models.load_model('cifar10.h5')
file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])


if file is None:
    st.text("Please upload an image file")
else:
    image_bytes = file.read()
    image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(image_array, 1)
    st.image(img, use_column_width=True) 
    img = cv2.resize(img, (32,32))
    img = img.astype('float32')
    img = img/255.0
    img = np.reshape(img, (1, 32, 32, 3))
    encoding = model.predict(img)  
    
    classes = ["airplane" , "automobile" , "bird", "cat" , "deer" , "dog" , "frog" , "horse" , "ship" , "truck"]
    bashorat = np.argmax(encoding)
    score = encoding[0][bashorat]
    st.write("Predict  : ",classes[bashorat])
    st.write("Score : ",score)
    
    
    
