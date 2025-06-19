# import matplotlib.pyplot as plt
# import pickle

# # Option 1: Save 'history' in model training script using pickle
# # with open('train_history.pkl', 'wb') as f:
# #     pickle.dump(history.history, f)

# # Option 2: Load the model history
# with open('vehicle_classifier_vgg16.h5', 'rb') as f:
#     history = pickle.load(f)

# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.plot(history['accuracy'], label='Train Acc')
# plt.plot(history['val_accuracy'], label='Val Acc')
# plt.legend()
# plt.title("Accuracy")

# plt.subplot(1, 2, 2)
# plt.plot(history['loss'], label='Train Loss')
# plt.plot(history['val_loss'], label='Val Loss')
# plt.legend()
# plt.title("Loss")

# plt.show()



import matplotlib.pyplot as plt

# Manually entered data from your training output
train_accuracy = [0.5523, 0.8190, 0.8319, 0.8400, 0.8597, 0.8676, 0.8887, 0.8829, 0.8784, 0.8838]
val_accuracy   = [0.9121, 0.9129, 0.9285, 0.9170, 0.9236, 0.9310, 0.9334, 0.9384, 0.9269, 0.9277]

train_loss = [1.8088, 0.5444, 0.4789, 0.4363, 0.3834, 0.3956, 0.3163, 0.3197, 0.3270, 0.3097]
val_loss   = [0.2905, 0.2670, 0.2138, 0.2446, 0.2181, 0.2009, 0.2098, 0.1877, 0.1938, 0.1944]

epochs = range(1, 11)

# Accuracy Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, 'b-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'g-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'r-', label='Training Loss')
plt.plot(epochs, val_loss, 'orange', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()
