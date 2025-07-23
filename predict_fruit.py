from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('fruit_classifier.h5')

# Load and preprocess the image
img = image.load_img('test.jpg', target_size=(100, 100))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
classes = ['apple', 'banana', 'orange']
predicted_class = classes[np.argmax(prediction)]

print(" Prediction:", predicted_class)
