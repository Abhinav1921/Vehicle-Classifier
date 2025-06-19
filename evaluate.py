import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load trained model
model = tf.keras.models.load_model('vehicle_classifier_vgg16.h5', compile=False)

# Recompile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define validation data path
val_dir = 'vehicle_data_split/val'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generator for validation set
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
results = model.evaluate(val_generator, verbose=1)
print(f"✅ Validation Loss: {results[0]:.4f}")
print(f"✅ Validation Accuracy: {results[1] * 100:.2f}%")