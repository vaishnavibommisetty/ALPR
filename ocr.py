import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import string
from typing import List, Dict, Optional, Tuple

class LicensePlateOCR:
    """
    Robust OCR system for license plate text extraction
    Optimized for Indian license plates and real-world conditions
    """
    
    def __init__(self):
        """Initialize OCR with optimized configuration"""
        # Optimized Tesseract configuration for license plates
        self.tesseract_config = '--oem 3 --psm 6'
        
        # Character whitelist for license plates
        self.char_whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # Common OCR character corrections for license plates
        self.char_corrections = {
            '0': 'O', 'O': '0',  # Zero/O confusion
            '1': 'I', 'I': '1',  # One/I confusion
            '5': 'S', 'S': '5',  # Five/S confusion
            '6': 'G', 'G': '6',  # Six/G confusion
            '8': 'B', 'B': '8',  # Eight/B confusion
            '2': 'Z', 'Z': '2',  # Two/Z confusion
        }
        
        print("OCR Engine initialized successfully")
    
    def extract_text(self, plate_image: np.ndarray) -> str:
        """
        Extract text from license plate image with comprehensive preprocessing
        
        Args:
            plate_image (np.ndarray): Cropped license plate image
            
        Returns:
            str: Extracted license plate text
        """
        try:
            if plate_image.size == 0:
                return "UNKNOWN"
            
            # Preprocess image for optimal OCR
            preprocessed_image = self._preprocess_plate_image(plate_image)
            
            # Extract text using Tesseract
            raw_text = pytesseract.image_to_string(
                preprocessed_image, 
                config=self.tesseract_config,
                lang='eng'
            )
            
            # Clean and validate extracted text
            cleaned_text = self._clean_text(raw_text)
            
            # Apply character corrections
            corrected_text = self._apply_corrections(cleaned_text)
            
            # Validate final result
            if self._is_valid_plate_text(corrected_text):
                return corrected_text
            else:
                return "UNKNOWN"
                
        except Exception as e:
            print(f"Error in text extraction: {e}")
            return "UNKNOWN"
    
    def _preprocess_plate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Comprehensive preprocessing for license plate OCR
        
        Args:
            image (np.ndarray): Input license plate image
            
        Returns:
            np.ndarray: Preprocessed image optimized for OCR
        """
        try:
            # Step 1: Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Step 2: Resize for better OCR (increase height)
            h, w = gray.shape
            if h < 50:
                scale_factor = 100 / h
                new_w = int(w * scale_factor)
                gray = cv2.resize(gray, (new_w, 100), interpolation=cv2.INTER_CUBIC)
            
            # Step 3: Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Step 4: Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)
            
            # Step 5: Adaptive thresholding for better text separation
            adaptive = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Step 6: Morphological operations to clean up text
            # Remove small noise
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
            
            # Connect broken characters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            # Step 7: Final noise removal
            kernel = np.ones((1, 1), np.uint8)
            final = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            return final
            
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return image
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text (str): Raw OCR text
            
        Returns:
            str: Cleaned text
        """
        try:
            # Remove whitespace and newlines
            text = text.strip().replace(' ', '').replace('\n', '')
            
            # Keep only alphanumeric characters (A-Z, 0-9)
            text = re.sub(r'[^A-Za-z0-9]', '', text)
            
            # Convert to uppercase
            text = text.upper()
            
            # Remove common OCR artifacts
            text = re.sub(r'[^\w]', '', text)
            
            return text
            
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return ""
    
    def _apply_corrections(self, text: str) -> str:
        """
        Apply common character corrections for license plates
        
        Args:
            text (str): Cleaned text
            
        Returns:
            str: Corrected text
        """
        try:
            corrected = text
            
            # Apply character substitutions based on context
            for old_char, new_char in self.char_corrections.items():
                # Only replace if it makes sense in context
                if old_char in corrected:
                    corrected = corrected.replace(old_char, new_char)
            
            return corrected
            
        except Exception as e:
            print(f"Error applying corrections: {e}")
            return text
    
    def _is_valid_plate_text(self, text: str) -> bool:
        """
        Validate if text is a valid license plate format
        
        Args:
            text (str): Text to validate
            
        Returns:
            bool: True if valid license plate format
        """
        try:
            # Check minimum length (license plates typically have 6-10 characters)
            if len(text) < 4 or len(text) > 12:
                return False
            
            # Check if text contains only allowed characters
            if not all(c in self.char_whitelist for c in text):
                return False
            
            # Check for common Indian license plate patterns
            # Pattern 1: XX12XX1234 (e.g., MH12AB1234)
            if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$', text):
                return True
            
            # Pattern 2: XX1XX1234 (e.g., MH1AB1234)
            if re.match(r'^[A-Z]{2}[0-9]{1}[A-Z]{2}[0-9]{4}$', text):
                return True
            
            # Pattern 3: XX12X1234 (e.g., MH12A1234)
            if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$', text):
                return True
            
            # Pattern 4: General alphanumeric pattern
            if re.match(r'^[A-Z]{2,4}[0-9]{2,4}$', text):
                return True
            
            # Pattern 5: Numbers followed by letters
            if re.match(r'^[0-9]{2}[A-Z]{2}[0-9]{4}$', text):
                return True
            
            # Pattern 6: Simple alphanumeric (fallback)
            if re.match(r'^[A-Z0-9]{6,10}$', text):
                # Check if it has both letters and numbers
                has_letters = any(c.isalpha() for c in text)
                has_numbers = any(c.isdigit() for c in text)
                return has_letters and has_numbers
            
            return False
            
        except Exception as e:
            print(f"Error validating plate text: {e}")
            return False
    
    def extract_text_with_confidence(self, plate_image: np.ndarray) -> Dict:
        """
        Extract text with confidence estimation
        
        Args:
            plate_image (np.ndarray): Cropped license plate image
            
        Returns:
            Dict: Text extraction result with confidence
        """
        try:
            # Extract text
            text = self.extract_text(plate_image)
            
            if text == "UNKNOWN":
                return {"text": "UNKNOWN", "confidence": 0.0}
            
            # Estimate confidence based on text characteristics
            confidence = self._estimate_confidence(text, plate_image)
            
            return {"text": text, "confidence": confidence}
            
        except Exception as e:
            print(f"Error in text extraction with confidence: {e}")
            return {"text": "UNKNOWN", "confidence": 0.0}
    
    def _estimate_confidence(self, text: str, image: np.ndarray) -> float:
        """
        Estimate confidence of OCR result
        
        Args:
            text (str): Extracted text
            image (np.ndarray): Original plate image
            
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        try:
            confidence = 0.5  # Base confidence
            
            # Length scoring (optimal length is 8-10 characters)
            if 8 <= len(text) <= 10:
                confidence += 0.2
            elif 6 <= len(text) <= 12:
                confidence += 0.1
            
            # Pattern matching scoring
            if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$', text):
                confidence += 0.3  # Perfect Indian plate pattern
            elif re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$', text):
                confidence += 0.2  # Good Indian plate pattern
            elif re.match(r'^[A-Z0-9]{6,10}$', text):
                confidence += 0.1  # General alphanumeric
            
            # Character quality scoring
            if all(c in self.char_whitelist for c in text):
                confidence += 0.1
            
            # Image quality scoring (if available)
            if image.size > 0:
                # Check image contrast
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                # Calculate contrast (standard deviation)
                contrast = np.std(gray)
                if contrast > 50:
                    confidence += 0.1
                elif contrast > 30:
                    confidence += 0.05
            
            return min(confidence, 0.95)  # Cap at 95%
            
        except Exception as e:
            print(f"Error estimating confidence: {e}")
            return 0.5

# Fallback OCR class for compatibility
class SimpleLicensePlateOCR:
    """
    Simple OCR fallback for compatibility
    """
    
    def __init__(self):
        self.ocr_engine = LicensePlateOCR()
    
    def extract_text(self, plate_image: np.ndarray) -> str:
        """Extract text using the robust OCR engine"""
        return self.ocr_engine.extract_text(plate_image)
