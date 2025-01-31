import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
cnn_model = load_model('C:/Users/Thomas Palero/Desktop/license_plate/cnn_license_plate_model.h5')

# Resize the image to match the model's expected input size
def preprocess_image(image):
    # Resize image to (128, 128, 3) to make sure it fits the model input
    resized_image = cv2.resize(image, (128, 128))  # Resizing for model input (height, width)
    # Normalize the image to have values between 0 and 1
    preprocessed_image = resized_image.astype('float32') / 255.0
    # Convert to the shape the model expects (batch size, height, width, channels)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    return preprocessed_image

# Function to detect plate number
def detect_plate_number(plate_region):
    # Preprocess the input image
    preprocessed_image = preprocess_image(plate_region)
    
    # Check the shape of the image before feeding to the model
    print("Preprocessed image shape:", preprocessed_image.shape)
    
    # Predict using the trained model
    predictions = cnn_model.predict(preprocessed_image)
    
    # Decode the predictions (assuming a label_encoder is available)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

# Load your input image (replace with actual image file path)
plate_image = cv2.imread('C:\\Users\\Thomas Palero\\Desktop\\license_plate\\dataset\\train\\3_jpg.rf.7bcdb5bd0e3b31f31b99b9ea9df14dd7.jpg')

# Call the function with the image region of interest (ROI)
plate_number = detect_plate_number(plate_image)

# Output the result
print("Predicted plate number class:", plate_number)
