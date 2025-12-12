import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = "reid_model.h5"
TEST_IMAGE_PATH = "/Users/sandevjayaweera/Downloads/Face Detection & Re-identification - Face Dataset/test/0.jpg"
IMAGE_SIZE = (128, 128)

model = tf.keras.models.load_model(MODEL_PATH)

img = image.load_img(TEST_IMAGE_PATH, target_size=IMAGE_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
predicted_class = np.argmax(pred)

print("Predicted Identity:", predicted_class)
print("Confidence:", np.max(pred))
