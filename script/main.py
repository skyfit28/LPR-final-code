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
            2: [5, 2],  # Wednesday
            3: [7, 8],  # Thursday
            4: [9, 0],  # Friday
            5: [],      # Saturday (no coding)
            6: []       # Sunday (no coding)
        }
        
        # Define coding hours (7:00 AM to 8:00 PM)
        self.coding_start_hour = 0
        self.coding_end_hour = 24
    
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
    """Handles license plate detection and recognition."""
    
    def __init__(self):
        self.plate_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        )
        # Set Tesseract configuration for license plates
        self.tesseract_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
    def preprocess_plate(self, plate_image: np.ndarray) -> np.ndarray:
        """Preprocess plate image for OCR."""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get black text on white background
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Remove noise
        blur = cv2.GaussianBlur(thresh, (3,3), 0)
        
        # Increase contrast
        contrast = cv2.convertScaleAbs(blur, alpha=1.5, beta=0)
        
        # Resize image to be larger (helps with OCR accuracy)
        return cv2.resize(contrast, (0, 0), fx=2, fy=2)
    
    def detect_plates(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect license plates in frame."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 50)
        )
        return plates
    
    def recognize_plate_number(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """Recognize characters in plate image using Tesseract OCR and calculate confidence."""
        try:
            # Preprocess the image
            processed_image = self.preprocess_plate(plate_image)
            
            # Perform OCR with confidence values
            ocr_result = pytesseract.image_to_data(
                processed_image,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract the OCR text and confidence scores
            plate_text = ''.join(ocr_result['text']).strip()
            confidences = [int(conf) for conf in ocr_result['conf'] if conf != '-1']
            
            # Calculate the average confidence
            if confidences:
                avg_confidence = np.mean(confidences)
            else:
                avg_confidence = 0.0
            
            # Validate the plate text (length between 5 and 8 characters is typical)
            if len(plate_text) >= 5 and len(plate_text) <= 8:
                return plate_text, avg_confidence
            return "", avg_confidence
    
        except Exception as e:
            logging.error(f"Error in plate recognition: {str(e)}")
            return "", 0.0

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
    # Initialize components
    plate_detector = PlateDetector()
    violation_recorder = ViolationRecorder()
    coding_scheme = CodingScheme()
    
    # Start video capture
    cap = cv2.VideoCapture(1)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Process every 3rd frame for better performance
            if frame_count % 3 != 0:
                continue
            
            # Detect plates
            plates = plate_detector.detect_plates(frame)
            
            for (x, y, w, h) in plates:
                # Extract plate region with some margin
                margin = 10
                y_start = max(y - margin, 0)
                y_end = min(y + h + margin, frame.shape[0])
                x_start = max(x - margin, 0)
                x_end = min(x + w + margin, frame.shape[1])
                plate_region = frame[y_start:y_end, x_start:x_end]
                
                # Recognize plate number and get confidence
                plate_number, confidence = plate_detector.recognize_plate_number(plate_region)
                
                if plate_number:
                    # Check if plate violates coding scheme
                    is_violated = coding_scheme.is_plate_violated(plate_number)
                    
                    # Set color based on violation status
                    color = (0, 0, 255) if is_violated else (0, 255, 0)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Display plate number and confidence
                    cv2.putText(frame, f"{plate_number} ({confidence:.2f}%)", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.8, color, 2)
                    
                    # If violated, record it
                    if is_violated:
                        if violation_recorder.record_violation(plate_number):
                            # Display "Recording Violation" text
                            cv2.putText(frame, "Recording Violation", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                      1, (0, 0, 255), 2)
            
            # Display current time and coding restrictions
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, f"Time: {current_time}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, (255, 255, 255), 2)
            
            # Display the frame
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
