import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import datetime
import pytesseract
import logging
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('license_plate_detection.log'),
        logging.StreamHandler()
    ]
)

class CodingScheme:
    """Handles number coding restrictions for Metro Manila."""
    
    def __init__(self):
        # Define coding scheme: {day_of_week: restricted_last_digits}
        self.coding_schedule = {
            0: [1, 2],  # Monday
            1: [3, 4],  # Tuesday
            2: [5, 6],  # Wednesday
            3: [7, 8],  # Thursday
            4: [9, 0],  # Friday
            5: [],      # Saturday (no coding)
            6: [1, 2]       # Sunday (no coding)
        }
        
        # Define coding hours (7:00 AM to 8:00 PM)
        self.coding_start_hour = 0
        self.coding_end_hour = 20
    
    def is_plate_violated(self, plate_number: str) -> bool:
        """Check if a plate number violates the current coding scheme."""
        now = datetime.now()
        day_of_week = now.weekday()
        current_hour = now.hour
        
        # Check if it's within coding hours
        if current_hour < self.coding_start_hour or current_hour >= self.coding_end_hour:
            return False
            
        # Check if it's a weekend
        if day_of_week in [5, ]:
            return False
            
        try:
            # Get the last digit of the plate number
            last_digit = int(plate_number[-1])
            
            # Check if the last digit is restricted today
            return last_digit in self.coding_schedule[day_of_week]
        except (IndexError, ValueError):
            logging.error(f"Invalid plate number format: {plate_number}")
            return False

class PlateDetector:
    """Updated with K-Means clustering and CNN recognition."""

    def __init__(self, cnn_model_path: str):
        self.cnn_model = load_model(cnn_model_path)  # Load your CNN model
        self.tesseract_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    def apply_kmeans(self, image: np.ndarray) -> np.ndarray:
        """Segment image using K-Means clustering."""
        # Reshape image for clustering
        reshaped_image = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=2, random_state=42).fit(reshaped_image)
        segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        return segmented_img.reshape(image.shape).astype(np.uint8)

    def preprocess_plate(self, plate_image: np.ndarray) -> np.ndarray:
        """Preprocess plate image for CNN."""
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 64))  # Resize for CNN input
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=(0, -1))

    def recognize_plate_number(self, plate_image: np.ndarray) -> str:
        """Recognize plate characters using CNN."""
        processed_image = self.preprocess_plate(plate_image)
        prediction = self.cnn_model.predict(processed_image)
        return ''.join(chr(np.argmax(char_prob)) for char_prob in prediction)


class ViolationRecorder:
    """Handles violation recording to Excel."""
    
    def __init__(self, output_path: str = 'violations.xlsx'):
        self.output_path = output_path
        self.recorded_plates = set()  # To avoid duplicate records
        self._initialize_excel()
    
    def _initialize_excel(self):
        """Initialize Excel file if it doesn't exist."""
        if not os.path.exists(self.output_path):
            df = pd.DataFrame(columns=['Plate Number', 'Date', 'Time', 'Violation Type'])
            df.to_excel(self.output_path, index=False)
            logging.info(f"Created new violations file: {self.output_path}")
    
    def record_violation(self, plate_number: str, violation_type: str = "Number Coding"):
        """Record violation if it hasn't been recorded recently."""
        if plate_number and plate_number not in self.recorded_plates:
            try:
                violation_data = {
                    'Plate Number': [plate_number],
                    'Date': [datetime.now().strftime('%Y-%m-%d')],
                    'Time': [datetime.now().strftime('%H:%M:%S')],
                    'Violation Type': [violation_type]
                }
                new_df = pd.DataFrame(violation_data)
                
                # Read existing data
                if os.path.exists(self.output_path):
                    existing_df = pd.read_excel(self.output_path)
                    # Concatenate new data
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    updated_df = new_df
                
                # Save to Excel
                updated_df.to_excel(self.output_path, index=False)
                self.recorded_plates.add(plate_number)
                logging.info(f"Recorded violation: {plate_number} - {violation_type}")
                return True
                
            except Exception as e:
                logging.error(f"Error recording violation: {str(e)}")
                return False
        return False

def main():
    # Load CNN model (ensure the model path is valid)
    plate_detector = PlateDetector(cnn_model_path='cnn_plate_recognition.h5')
    violation_recorder = ViolationRecorder()
    coding_scheme = CodingScheme()

    cap = cv2.VideoCapture(0)
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 3 != 0:
                continue

            # Apply K-Means for segmentation
            clustered_frame = plate_detector.apply_kmeans(frame)

            # Detect plates (still using Haar cascades for simplicity)
            plates = plate_detector.detect_plates(clustered_frame)

            for (x, y, w, h) in plates:
                plate_region = frame[y:y+h, x:x+w]
                plate_number = plate_detector.recognize_plate_number(plate_region)

                if plate_number:
                    is_violated = coding_scheme.is_plate_violated(plate_number)
                    color = (0, 0, 255) if is_violated else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, plate_number, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if is_violated:
                        violation_recorder.record_violation(plate_number)

            cv2.imshow('License Plate Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()