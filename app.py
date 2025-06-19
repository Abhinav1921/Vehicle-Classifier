import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

model = tf.keras.models.load_model('vehicle_classifier_vgg16.h5')
class_names = sorted(os.listdir('vehicle_data_split/train'))

st.title("🚗 Vehicle Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted: **{predicted_class}**")
