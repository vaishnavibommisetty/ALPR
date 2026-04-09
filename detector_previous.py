import cv2
import numpy as np
from PIL import Image
import easyocr
import torch
import re
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import os

class ComprehensiveLicensePlateDetector:
    """
    Multi-Method License Plate Detection System
    Text-First Detection with Contour and Edge-based fallbacks
    """
    
    def __init__(self):
        """Initialize detector with EasyOCR and CV methods"""
        try:
            # Initialize EasyOCR reader
            self.reader = easyocr.Reader(
                ['en'], 
                gpu=torch.cuda.is_available(),
                verbose=False
            )
            
            # Detection parameters
            self.confidence_threshold = 0.5
            self.processing_size = (640, 480)
            self.detection_cache = {}  # Cache for duplicate prevention
            
            # License plate validation patterns
            self.plate_patterns = [
                r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',  # MH12AB1234
                r'^[A-Z]{2}[0-9]{1}[A-Z]{2}[0-9]{4}$',   # MH1AB1234
                r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$',   # MH12A1234
                r'^[A-Z]{3}[0-9]{3}$',                     # ABC123
                r'^[A-Z]{2}[0-9]{4}$',                     # AB1234
                r'^[A-Z]{3}[0-9]{4}$',                     # ABC1234
                r'^[0-9]{2}[A-Z]{2}[0-9]{4}$',          # 12AB1234
                r'^[A-Z]{1,3}[0-9]{1,4}$',              # A123, AB123, ABC123
                r'^[A-Z0-9]{6,10}$'                      # General alphanumeric
            ]
            
            print("Comprehensive License Plate Detector initialized successfully")
            print(f"GPU Available: {torch.cuda.is_available()}")
            
        except Exception as e:
            print(f"Error initializing detector: {e}")
            self.reader = None
    
    def detect_license_plate(self, image_path: str) -> Dict:
        """
        Multi-method license plate detection pipeline
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Dict: Detection result with plate number and confidence
        """
        try:
            start_time = time.time()
            
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {"plate_number": "NOT_DETECTED", "confidence": 0, "error": "Invalid image"}
            
            # Check for duplicates within 5-minute window
            image_hash = self._get_image_hash(image)
            if self._is_duplicate(image_hash):
                return {"plate_number": "DUPLICATE", "confidence": 0, "error": "Duplicate detection"}
            
            # Resize for performance
            processed_image = self._preprocess_image(image)
            
            # Multi-method detection
            all_detections = []
            
            # Method 1: Text-First Detection (Priority 1)
            text_detections = self._text_first_detection(processed_image)
            for detection in text_detections:
                detection['method'] = 'text_first'
                detection['weight'] = 1.0
                all_detections.append(detection)
            
            # Method 2: Contour-Based Detection (Priority 2)
            contour_detections = self._contour_based_detection(processed_image)
            for detection in contour_detections:
                detection['method'] = 'contour_based'
                detection['weight'] = 0.8
                all_detections.append(detection)
            
            # Method 3: Edge-Based Detection (Priority 3)
            edge_detections = self._edge_based_detection(processed_image)
            for detection in edge_detections:
                detection['method'] = 'edge_based'
                detection['weight'] = 0.7
                all_detections.append(detection)
            
            # Select best result
            best_detection = self._select_best_detection(all_detections, image)
            
            if best_detection:
                # Cache the detection
                self._cache_detection(image_hash, best_detection)
                
                # Draw bounding box
                annotated_image = self._draw_detection(image, best_detection)
                
                # Save annotated image
                output_path = image_path.replace('.', '_detected.')
                cv2.imwrite(output_path, annotated_image)
                
                processing_time = time.time() - start_time
                
                return {
                    "plate_number": best_detection['text'],
                    "confidence": best_detection['final_confidence'],
                    "bbox": best_detection['bbox'],
                    "method": best_detection['method'],
                    "processing_time": processing_time,
                    "annotated_image_path": output_path,
                    "detection_count": len(all_detections)
                }
            else:
                return {
                    "plate_number": "NOT_DETECTED", 
                    "confidence": 0, 
                    "processing_time": time.time() - start_time,
                    "detection_count": 0,
                    "error": "No valid license plates found"
                }
                
        except Exception as e:
            print(f"Error in license plate detection: {e}")
            return {
                "plate_number": "PROCESSING_ERROR", 
                "confidence": 0, 
                "error": str(e)
            }
    
    def detect_license_plate_with_ocr(self, image_path: str, ocr_engine=None) -> Dict:
        """
        Detection with OCR integration (for compatibility)
        
        Args:
            image_path (str): Path to image file
            ocr_engine: OCR engine (for compatibility)
            
        Returns:
            Dict: Detection result
        """
        # Use built-in EasyOCR, ignore external OCR engine
        return self.detect_license_plate(image_path)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Comprehensive image preprocessing for detection
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            # Resize for performance
            h, w = image.shape[:2]
            if h > self.processing_size[1] or w > self.processing_size[0]:
                scale = min(self.processing_size[0]/w, self.processing_size[1]/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # LAB color space conversion for better contrast
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Bilateral filtering for noise reduction
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return filtered
            
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return image
    
    def _text_first_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Text-first detection using EasyOCR
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict]: Text-based detections
        """
        try:
            if self.reader is None:
                return []
            
            # Use EasyOCR to find text regions
            results = self.reader.readtext(image)
            
            detections = []
            
            for (bbox, text, confidence) in results:
                # Clean and validate text
                cleaned_text = self._clean_text(text)
                
                if self._is_valid_plate_text(cleaned_text):
                    # Convert bbox format
                    x1 = int(min([point[0] for point in bbox]))
                    y1 = int(min([point[1] for point in bbox]))
                    x2 = int(max([point[0] for point in bbox]))
                    y2 = int(max([point[1] for point in bbox]))
                    
                    # Expand bounding box to include full plate region
                    expanded_bbox = self._expand_bbox((x1, y1, x2, y2), image.shape)
                    
                    detections.append({
                        'text': cleaned_text,
                        'confidence': confidence,
                        'bbox': expanded_bbox,
                        'raw_text': text
                    })
            
            return detections
            
        except Exception as e:
            print(f"Error in text-first detection: {e}")
            return []
    
    def _contour_based_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Contour-based detection with OCR validation
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict]: Contour-based detections
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Canny edge detection
            edged = cv2.Canny(bilateral, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Aspect ratio and area validation
                aspect_ratio = w / h
                area = cv2.contourArea(contour)
                image_area = image.shape[0] * image.shape[1]
                area_ratio = area / image_area
                
                if (2.0 <= aspect_ratio <= 6.0 and 
                    0.001 <= area_ratio <= 0.1 and 
                    w > 60 and h > 15):
                    
                    # Extract region
                    plate_region = image[y:y+h, x:x+w]
                    
                    # Use EasyOCR to validate text
                    if self.reader:
                        ocr_results = self.reader.readtext(plate_region)
                        
                        for (bbox, text, confidence) in ocr_results:
                            cleaned_text = self._clean_text(text)
                            
                            if self._is_valid_plate_text(cleaned_text):
                                detections.append({
                                    'text': cleaned_text,
                                    'confidence': confidence * 0.8,  # Weight for contour method
                                    'bbox': (x, y, x + w, y + h),
                                    'raw_text': text
                                })
            
            return detections
            
        except Exception as e:
            print(f"Error in contour-based detection: {e}")
            return []
    
    def _edge_based_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Edge-based detection with morphological operations
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict]: Edge-based detections
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multiple morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Method 1: Morphological gradient
            morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            _, binary = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 2: Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Combine methods
            combined = cv2.bitwise_or(binary, adaptive)
            
            # Find contours
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Validation
                aspect_ratio = w / h
                area = cv2.contourArea(contour)
                
                if (1.5 <= aspect_ratio <= 8.0 and 
                    1000 <= area <= 50000 and 
                    w > 50 and h > 10):
                    
                    # Extract region
                    plate_region = image[y:y+h, x:x+w]
                    
                    # Use EasyOCR for text extraction
                    if self.reader:
                        ocr_results = self.reader.readtext(plate_region)
                        
                        for (bbox, text, confidence) in ocr_results:
                            cleaned_text = self._clean_text(text)
                            
                            if self._is_valid_plate_text(cleaned_text):
                                detections.append({
                                    'text': cleaned_text,
                                    'confidence': confidence * 0.7,  # Weight for edge method
                                    'bbox': (x, y, x + w, y + h),
                                    'raw_text': text
                                })
            
            return detections
            
        except Exception as e:
            print(f"Error in edge-based detection: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        try:
            # Remove non-alphanumeric characters
            cleaned = re.sub(r'[^A-Za-z0-9]', '', text)
            
            # Convert to uppercase
            cleaned = cleaned.upper()
            
            # Remove common OCR errors
            cleaned = re.sub(r'[^\w]', '', cleaned)
            
            return cleaned.strip()
            
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return ""
    
    def _is_valid_plate_text(self, text: str) -> bool:
        """
        Validate license plate text
        
        Args:
            text (str): Text to validate
            
        Returns:
            bool: True if valid license plate format
        """
        try:
            if not text or len(text) < 3 or len(text) > 15:
                return False
            
            # Check if it has mix of letters and numbers OR minimum 4 alphanumeric chars
            has_letters = any(c.isalpha() for c in text)
            has_numbers = any(c.isdigit() for c in text)
            
            if not (has_letters or has_numbers):
                return False
            
            # Check against patterns
            for pattern in self.plate_patterns:
                if re.match(pattern, text):
                    return True
            
            # General alphanumeric check
            if re.match(r'^[A-Z0-9]{4,10}$', text):
                return True
            
            return False
            
        except Exception as e:
            print(f"Error validating plate text: {e}")
            return False
    
    def _expand_bbox(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Expand bounding box to include full plate region
        
        Args:
            bbox: Original bounding box (x1, y1, x2, y2)
            image_shape: Image shape (h, w, c)
            
        Returns:
            Tuple[int, int, int, int]: Expanded bounding box
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = image_shape[:2]
            
            # Expand by 10% in each direction
            expand_x = int((x2 - x1) * 0.1)
            expand_y = int((y2 - y1) * 0.1)
            
            x1 = max(0, x1 - expand_x)
            y1 = max(0, y1 - expand_y)
            x2 = min(w, x2 + expand_x)
            y2 = min(h, y2 + expand_y)
            
            return (x1, y1, x2, y2)
            
        except Exception as e:
            print(f"Error expanding bbox: {e}")
            return bbox
    
    def _select_best_detection(self, detections: List[Dict], image: np.ndarray) -> Optional[Dict]:
        """
        Select best detection from multiple methods
        
        Args:
            detections: List of all detections
            image: Original image
            
        Returns:
            Dict: Best detection or None
        """
        try:
            if not detections:
                return None
            
            # Calculate final confidence for each detection
            for detection in detections:
                base_confidence = detection['confidence']
                weight = detection['weight']
                detection['final_confidence'] = base_confidence * weight
                
                # Additional quality scoring
                bbox = detection['bbox']
                quality_score = self._calculate_quality_score(bbox, detection['text'], image)
                detection['final_confidence'] *= quality_score
            
            # Select best detection
            best_detection = max(detections, key=lambda x: x['final_confidence'])
            
            return best_detection
            
        except Exception as e:
            print(f"Error selecting best detection: {e}")
            return None
    
    def _calculate_quality_score(self, bbox: Tuple[int, int, int, int], text: str, image: np.ndarray) -> float:
        """
        Calculate quality score for detection
        
        Args:
            bbox: Bounding box
            text: Detected text
            image: Original image
            
        Returns:
            float: Quality score (0.0 to 1.0)
        """
        try:
            score = 1.0
            
            # Text length scoring
            if 6 <= len(text) <= 10:
                score *= 1.2
            elif 4 <= len(text) <= 12:
                score *= 1.1
            
            # Aspect ratio scoring
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            aspect_ratio = w / h
            
            if 2.0 <= aspect_ratio <= 4.0:
                score *= 1.1
            elif 1.5 <= aspect_ratio <= 6.0:
                score *= 1.05
            
            # Position scoring (plates usually in lower half)
            rel_y = y1 / image.shape[0]
            if 0.4 <= rel_y <= 0.9:
                score *= 1.05
            
            return min(score, 1.5)  # Cap at 1.5
            
        except Exception as e:
            print(f"Error calculating quality score: {e}")
            return 1.0
    
    def _draw_detection(self, image: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Draw detection on image
        
        Args:
            image: Original image
            detection: Detection result
            
        Returns:
            np.ndarray: Annotated image
        """
        try:
            annotated = image.copy()
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['final_confidence']
            method = detection['method']
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence > 0.6:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection['text']} ({confidence:.2f}) [{method}]"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                          (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"Error drawing detection: {e}")
            return image
    
    def _get_image_hash(self, image: np.ndarray) -> str:
        """
        Get image hash for duplicate detection
        
        Args:
            image: Input image
            
        Returns:
            str: Image hash
        """
        try:
            # Simple hash based on image dimensions and mean
            h, w = image.shape[:2]
            mean_val = np.mean(image)
            return f"{h}_{w}_{mean_val:.0f}"
            
        except Exception as e:
            print(f"Error getting image hash: {e}")
            return ""
    
    def _is_duplicate(self, image_hash: str) -> bool:
        """
        Check if image is duplicate within 5-minute window
        
        Args:
            image_hash: Image hash
            
        Returns:
            bool: True if duplicate
        """
        try:
            current_time = datetime.now()
            
            if image_hash in self.detection_cache:
                last_detection = self.detection_cache[image_hash]['timestamp']
                time_diff = current_time - last_detection
                
                if time_diff < timedelta(minutes=5):
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking duplicate: {e}")
            return False
    
    def _cache_detection(self, image_hash: str, detection: Dict):
        """
        Cache detection result
        
        Args:
            image_hash: Image hash
            detection: Detection result
        """
        try:
            self.detection_cache[image_hash] = {
                'detection': detection,
                'timestamp': datetime.now()
            }
            
            # Clean old cache entries (older than 5 minutes)
            current_time = datetime.now()
            expired_keys = []
            
            for key, value in self.detection_cache.items():
                if current_time - value['timestamp'] > timedelta(minutes=5):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.detection_cache[key]
                
        except Exception as e:
            print(f"Error caching detection: {e}")
    
    def detect_from_pil_image(self, pil_image: Image.Image) -> Dict:
        """
        Detect license plate from PIL Image
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Dict: Detection result
        """
        try:
            # Convert PIL to OpenCV format
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Save temporary image
            temp_path = "temp_detection.jpg"
            cv2.imwrite(temp_path, image)
            
            # Detect using main method
            result = self.detect_license_plate(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
            
        except Exception as e:
            print(f"Error in PIL image detection: {e}")
            return {"plate_number": "NOT_DETECTED", "confidence": 0, "error": str(e)}

# Fallback detector for compatibility
class SimpleLicensePlateDetector:
    """
    Simple fallback detector for compatibility
    """
    
    def __init__(self):
        self.detector = ComprehensiveLicensePlateDetector()
    
    def detect_license_plate(self, image_path: str) -> Dict:
        """Use comprehensive detector"""
        return self.detector.detect_license_plate(image_path)
    
    def detect_license_plate_with_ocr(self, image_path: str, ocr_engine=None) -> Dict:
        """Use comprehensive detector"""
        return self.detector.detect_license_plate_with_ocr(image_path, ocr_engine)
    
    def detect_from_pil_image(self, pil_image: Image.Image) -> Dict:
        """Use comprehensive detector"""
        return self.detector.detect_from_pil_image(pil_image)
