import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

model = tf.keras.models.load_model('vehicle_classifier_vgg16.h5')
class_names = sorted(os.listdir('vehicle_data_split/train'))

def predict_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    print(f"✅ Predicted Class: {predicted_class}")

# Test
predict_img("C:/Users/agraw/Downloads/download (1).jpeg")
