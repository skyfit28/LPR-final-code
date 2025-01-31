import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Load Dataset and Annotations
annotations = pd.read_csv('C:/Users/Thomas Palero/Desktop/license_plate/dataset/annotations/train_annotations.csv')  # Replace with your actual CSV file

images = []
labels = []

for idx, row in annotations.iterrows():
    img_path = os.path.join('C:/Users/Thomas Palero/Desktop/license_plate/dataset/train', row['filename'])  # Replace with your dataset folder
    img = tf.keras.utils.load_img(img_path, target_size=(128, 128))  # Resize to (128, 128)
    img_array = tf.keras.utils.img_to_array(img)
    images.append(img_array)
    labels.append(row['label'])

images = np.array(images) / 255.0  # Normalize pixel values

# Step 2: Encode Labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)  # Convert to one-hot encoding

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)

# Step 4: Define the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # Input shape
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    GlobalAveragePooling2D(),  # Replaces Flatten, automatically handles feature map size
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer for classification
])

# Step 5: Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Data Augmentation (Optional)
data_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
data_gen.fit(X_train)

# Step 7: Train the Model
history = model.fit(data_gen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=10)  # Adjust epochs as needed

# Step 8: Save the Trained Model
model.save('cnn_license_plate_model.h5')

# Step 9: Load the Model for Prediction
cnn_model = tf.keras.models.load_model('cnn_license_plate_model.h5')

# Example Prediction (Replace 'sample_image.jpg' with an actual test image)
sample_image = tf.keras.utils.load_img('dataset/train/3_jpg.rf.7bcdb5bd0e3b31f31b99b9ea9df14dd7.jpg', target_size=(128, 128))  # Update target size
sample_image_array = tf.keras.utils.img_to_array(sample_image) / 255.0
sample_image_array = np.expand_dims(sample_image_array, axis=0)  # Add batch dimension
predictions = cnn_model.predict(sample_image_array)
predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])
print("Predicted Label:", predicted_label[0])
