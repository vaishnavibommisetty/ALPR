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

class AggressiveLicensePlateDetector:
    """
    Aggressive Text-First License Plate Detection
    Prioritizes finding text regions first, then validates as plates
    """
    
    def __init__(self):
        """Initialize detector with aggressive text-first approach"""
        try:
            # Initialize EasyOCR reader with permissive settings
            self.reader = easyocr.Reader(
                ['en'], 
                gpu=torch.cuda.is_available(),
                verbose=False
            )
            
            # Aggressive detection parameters
            self.confidence_threshold = 0.1  # Very low threshold
            self.processing_size = (800, 600)  # Larger processing size
            self.detection_cache = {}
            
            # Very permissive license plate patterns
            self.plate_patterns = [
                r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',  # MH12AB1234
                r'^[A-Z]{2}[0-9]{1}[A-Z]{2}[0-9]{4}$',   # MH1AB1234
                r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$',   # MH12A1234
                r'^[A-Z]{3}[0-9]{3}$',                     # ABC123
                r'^[A-Z]{2}[0-9]{4}$',                     # AB1234
                r'^[A-Z]{3}[0-9]{4}$',                     # ABC1234
                r'^[0-9]{2}[A-Z]{2}[0-9]{4}$',          # 12AB1234
                r'^[A-Z]{1,3}[0-9]{1,4}$',              # A123, AB123, ABC123
                r'^[A-Z0-9]{3,12}$'                      # Very permissive alphanumeric
            ]
            
            print("Aggressive Text-First License Plate Detector initialized successfully")
            print(f"GPU Available: {torch.cuda.is_available()}")
            
        except Exception as e:
            print(f"Error initializing aggressive detector: {e}")
            self.reader = None
    
    def detect_license_plate(self, image_path: str) -> Dict:
        """
        Aggressive license plate detection - text-first approach
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Dict: Detection result with plate number and confidence
        """
        try:
            start_time = time.time()
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"plate_number": "NOT_DETECTED", "confidence": 0, "error": "Invalid image"}
            
            # Check for duplicates
            image_hash = self._get_image_hash(image)
            if self._is_duplicate(image_hash):
                return {"plate_number": "DUPLICATE", "confidence": 0, "error": "Duplicate detection"}
            
            # Aggressive preprocessing
            processed_image = self._aggressive_preprocessing(image)
            
            # Multiple text detection attempts
            all_detections = []
            
            # Method 1: Direct EasyOCR on original image
            direct_detections = self._aggressive_text_detection(processed_image)
            for detection in direct_detections:
                detection['method'] = 'direct_text'
                detection['weight'] = 1.0
                all_detections.append(detection)
            
            # Method 2: EasyOCR on enhanced contrast
            enhanced_detections = self._enhanced_contrast_text_detection(processed_image)
            for detection in enhanced_detections:
                detection['method'] = 'enhanced_contrast'
                detection['weight'] = 0.9
                all_detections.append(detection)
            
            # Method 3: EasyOCR on thresholded regions
            threshold_detections = self._threshold_text_detection(processed_image)
            for detection in threshold_detections:
                detection['method'] = 'threshold_text'
                detection['weight'] = 0.8
                all_detections.append(detection)
            
            # Method 4: Contour-based text extraction
            contour_detections = self._contour_text_extraction(processed_image)
            for detection in contour_detections:
                detection['method'] = 'contour_text'
                detection['weight'] = 0.7
                all_detections.append(detection)
            
            # Method 5: Grid-based text search
            grid_detections = self._grid_text_search(processed_image)
            for detection in grid_detections:
                detection['method'] = 'grid_text'
                detection['weight'] = 0.6
                all_detections.append(detection)
            
            # Select best result
            best_detection = self._select_best_aggressive_detection(all_detections, image)
            
            if best_detection:
                # Cache detection
                self._cache_detection(image_hash, best_detection)
                
                # Draw detection
                annotated_image = self._draw_aggressive_detection(image, best_detection)
                
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
                    "error": "No text regions found"
                }
                
        except Exception as e:
            print(f"Error in aggressive license plate detection: {e}")
            return {
                "plate_number": "PROCESSING_ERROR", 
                "confidence": 0, 
                "error": str(e)
            }
    
    def detect_license_plate_with_ocr(self, image_path: str, ocr_engine=None) -> Dict:
        """Detection with OCR integration (for compatibility)"""
        return self.detect_license_plate(image_path)
    
    def _aggressive_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Aggressive preprocessing to enhance text visibility
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Aggressively preprocessed image
        """
        try:
            # Convert to LAB for better contrast
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Very aggressive CLAHE
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
            l = clahe.apply(l)
            
            # Merge back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter
            bilateral = cv2.bilateralFilter(gray, 15, 25, 25)
            
            return bilateral
            
        except Exception as e:
            print(f"Error in aggressive preprocessing: {e}")
            return image
    
    def _aggressive_text_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Aggressive text detection using EasyOCR with permissive settings
        
        Args:
            image (np.ndarray): Preprocessed image
            
        Returns:
            List[Dict]: Aggressive text detections
        """
        try:
            if self.reader is None:
                return []
            
            # Convert back to BGR for EasyOCR
            if len(image.shape) == 2:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image
            
            # Use EasyOCR with very permissive settings
            results = self.reader.readtext(image_bgr)
            
            detections = []
            
            for (bbox, text, confidence) in results:
                # Very permissive text cleaning
                cleaned_text = self._aggressive_text_cleaning(text)
                
                # Very permissive validation
                if self._is_permissive_plate_text(cleaned_text):
                    # Convert bbox format
                    x1 = int(min([point[0] for point in bbox]))
                    y1 = int(min([point[1] for point in bbox]))
                    x2 = int(max([point[0] for point in bbox]))
                    y2 = int(max([point[1] for point in bbox]))
                    
                    # Aggressive bounding box expansion
                    expanded_bbox = self._aggressive_bbox_expansion((x1, y1, x2, y2), image.shape)
                    
                    detections.append({
                        'text': cleaned_text,
                        'confidence': confidence,
                        'bbox': expanded_bbox,
                        'raw_text': text
                    })
            
            return detections
            
        except Exception as e:
            print(f"Error in aggressive text detection: {e}")
            return []
    
    def _enhanced_contrast_text_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Text detection with enhanced contrast
        
        Args:
            image (np.ndarray): Preprocessed image
            
        Returns:
            List[Dict]: Enhanced contrast text detections
        """
        try:
            # Apply histogram equalization
            equalized = cv2.equalizeHist(image)
            
            # Apply adaptive threshold
            adaptive = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
            
            # Convert back to BGR for EasyOCR
            image_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
            
            if self.reader:
                results = self.reader.readtext(image_bgr)
                
                detections = []
                for (bbox, text, confidence) in results:
                    cleaned_text = self._aggressive_text_cleaning(text)
                    
                    if self._is_permissive_plate_text(cleaned_text):
                        x1 = int(min([point[0] for point in bbox]))
                        y1 = int(min([point[1] for point in bbox]))
                        x2 = int(max([point[0] for point in bbox]))
                        y2 = int(max([point[1] for point in bbox]))
                        
                        expanded_bbox = self._aggressive_bbox_expansion((x1, y1, x2, y2), image.shape)
                        
                        detections.append({
                            'text': cleaned_text,
                            'confidence': confidence * 0.9,  # Slightly lower confidence
                            'bbox': expanded_bbox,
                            'raw_text': text
                        })
                
                return detections
            
            return []
            
        except Exception as e:
            print(f"Error in enhanced contrast text detection: {e}")
            return []
    
    def _threshold_text_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Text detection using multiple threshold methods
        
        Args:
            image (np.ndarray): Preprocessed image
            
        Returns:
            List[Dict]: Threshold text detections
        """
        try:
            detections = []
            
            # Multiple threshold methods
            threshold_methods = [
                ('binary', cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]),
                ('otsu', cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                ('adaptive', cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
            ]
            
            for method_name, thresholded in threshold_methods:
                if self.reader:
                    # Convert to BGR for EasyOCR
                    image_bgr = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
                    
                    results = self.reader.readtext(image_bgr)
                    
                    for (bbox, text, confidence) in results:
                        cleaned_text = self._aggressive_text_cleaning(text)
                        
                        if self._is_permissive_plate_text(cleaned_text):
                            x1 = int(min([point[0] for point in bbox]))
                            y1 = int(min([point[1] for point in bbox]))
                            x2 = int(max([point[0] for point in bbox]))
                            y2 = int(max([point[1] for point in bbox]))
                            
                            expanded_bbox = self._aggressive_bbox_expansion((x1, y1, x2, y2), image.shape)
                            
                            detections.append({
                                'text': cleaned_text,
                                'confidence': confidence * 0.8,  # Lower confidence for threshold method
                                'bbox': expanded_bbox,
                                'raw_text': text,
                                'threshold_method': method_name
                            })
            
            return detections
            
        except Exception as e:
            print(f"Error in threshold text detection: {e}")
            return []
    
    def _contour_text_extraction(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text from contour regions
        
        Args:
            image (np.ndarray): Preprocessed image
            
        Returns:
            List[Dict]: Contour text detections
        """
        try:
            # Find contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Very permissive validation
                if w > 20 and h > 10 and w * h > 200:
                    # Extract region
                    if y + h <= image.shape[0] and x + w <= image.shape[1]:
                        region = image[y:y+h, x:x+w]
                        
                        # Use EasyOCR on region
                        if self.reader:
                            # Convert to BGR for EasyOCR
                            region_bgr = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
                            
                            results = self.reader.readtext(region_bgr)
                            
                            for (bbox, text, confidence) in results:
                                cleaned_text = self._aggressive_text_cleaning(text)
                                
                                if self._is_permissive_plate_text(cleaned_text):
                                    detections.append({
                                        'text': cleaned_text,
                                        'confidence': confidence * 0.7,  # Lower confidence for contour method
                                        'bbox': (x, y, x + w, y + h),
                                        'raw_text': text
                                    })
            
            return detections
            
        except Exception as e:
            print(f"Error in contour text extraction: {e}")
            return []
    
    def _grid_text_search(self, image: np.ndarray) -> List[Dict]:
        """
        Search for text in grid regions
        
        Args:
            image (np.ndarray): Preprocessed image
            
        Returns:
            List[Dict]: Grid text detections
        """
        try:
            detections = []
            h, w = image.shape
            
            # Define grid regions
            grid_size = 4
            for i in range(grid_size):
                for j in range(grid_size):
                    x1 = int(j * w / grid_size)
                    y1 = int(i * h / grid_size)
                    x2 = int((j + 1) * w / grid_size)
                    y2 = int((i + 1) * h / grid_size)
                    
                    # Extract grid region
                    region = image[y1:y2, x1:x2]
                    
                    if region.size > 0 and self.reader:
                        # Convert to BGR for EasyOCR
                        region_bgr = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
                        
                        results = self.reader.readtext(region_bgr)
                        
                        for (bbox, text, confidence) in results:
                            cleaned_text = self._aggressive_text_cleaning(text)
                            
                            if self._is_permissive_plate_text(cleaned_text):
                                # Adjust coordinates to full image
                                full_x1 = x1 + int(min([point[0] for point in bbox]))
                                full_y1 = y1 + int(min([point[1] for point in bbox]))
                                full_x2 = x1 + int(max([point[0] for point in bbox]))
                                full_y2 = y1 + int(max([point[1] for point in bbox]))
                                
                                expanded_bbox = self._aggressive_bbox_expansion((full_x1, full_y1, full_x2, full_y2), image.shape)
                                
                                detections.append({
                                    'text': cleaned_text,
                                    'confidence': confidence * 0.6,  # Lower confidence for grid method
                                    'bbox': expanded_bbox,
                                    'raw_text': text,
                                    'grid_position': f"{i},{j}"
                                })
            
            return detections
            
        except Exception as e:
            print(f"Error in grid text search: {e}")
            return []
    
    def _aggressive_text_cleaning(self, text: str) -> str:
        """
        Very permissive text cleaning
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Aggressively cleaned text
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
            print(f"Error in aggressive text cleaning: {e}")
            return ""
    
    def _is_permissive_plate_text(self, text: str) -> bool:
        """
        Very permissive validation for license plate text
        
        Args:
            text (str): Text to validate
            
        Returns:
            bool: True if text could be a license plate
        """
        try:
            # Very permissive validation
            if not text or len(text) < 2 or len(text) > 15:
                return False
            
            # Must have at least one alphanumeric character
            if not re.search(r'[A-Za-z0-9]', text):
                return False
            
            # Check against permissive patterns
            for pattern in self.plate_patterns:
                if re.match(pattern, text):
                    return True
            
            # Very permissive alphanumeric check
            if re.match(r'^[A-Z0-9]{2,12}$', text):
                return True
            
            return False
            
        except Exception as e:
            print(f"Error in permissive plate text validation: {e}")
            return False
    
    def _aggressive_bbox_expansion(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Very aggressive bounding box expansion
        
        Args:
            bbox: Original bounding box (x1, y1, x2, y2)
            image_shape: Image shape (h, w, c)
            
        Returns:
            Tuple[int, int, int, int]: Aggressively expanded bounding box
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = image_shape[:2]
            
            # Calculate expansion
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Very aggressive expansion
            expand_x = max(20, int(bbox_width * 0.25))
            expand_y = max(20, int(bbox_height * 0.25))
            
            # Apply expansion with boundaries
            x1 = max(0, x1 - expand_x)
            y1 = max(0, y1 - expand_y)
            x2 = min(w, x2 + expand_x)
            y2 = min(h, y2 + expand_y)
            
            return (x1, y1, x2, y2)
            
        except Exception as e:
            print(f"Error in aggressive bbox expansion: {e}")
            return bbox
    
    def _select_best_aggressive_detection(self, detections: List[Dict], image: np.ndarray) -> Optional[Dict]:
        """
        Select best detection from aggressive methods
        
        Args:
            detections: List of all detections
            image: Original image
            
        Returns:
            Dict: Best detection or None
        """
        try:
            if not detections:
                return None
            
            # Calculate aggressive confidence for each detection
            for detection in detections:
                base_confidence = detection['confidence']
                weight = detection.get('weight', 0.6)
                detection['final_confidence'] = base_confidence * weight
                
                # Quality scoring
                bbox = detection['bbox']
                quality_score = self._calculate_aggressive_quality_score(bbox, detection['text'], image, detection.get('method', ''))
                detection['final_confidence'] *= quality_score
            
            # Select best detection
            best_detection = max(detections, key=lambda x: x['final_confidence'])
            
            return best_detection
            
        except Exception as e:
            print(f"Error selecting best aggressive detection: {e}")
            return None
    
    def _calculate_aggressive_quality_score(self, bbox: Tuple[int, int, int, int], text: str, image: np.ndarray, method: str) -> float:
        """
        Calculate aggressive quality score
        
        Args:
            bbox: Bounding box
            text: Detected text
            image: Original image
            method: Detection method
            
        Returns:
            float: Aggressive quality score (0.0 to 2.0)
        """
        try:
            score = 1.0
            
            # Method-based scoring
            if 'direct_text' in method:
                score *= 1.5  # Highest boost for direct text
            elif 'enhanced_contrast' in method:
                score *= 1.3  # High boost for enhanced contrast
            elif 'threshold' in method:
                score *= 1.2  # Medium boost for threshold
            elif 'contour' in method:
                score *= 1.1  # Small boost for contour
            elif 'grid' in method:
                score *= 1.0  # No boost for grid
            
            # Text length scoring
            if 4 <= len(text) <= 8:
                score *= 1.3
            elif 3 <= len(text) <= 10:
                score *= 1.2
            elif 2 <= len(text) <= 12:
                score *= 1.1
            
            # Aspect ratio scoring
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            aspect_ratio = w / h
            
            if 1.5 <= aspect_ratio <= 6.0:
                score *= 1.2
            elif 1.0 <= aspect_ratio <= 8.0:
                score *= 1.1
            
            # Position scoring
            rel_y = y1 / image.shape[0]
            if 0.2 <= rel_y <= 0.9:
                score *= 1.1
            
            # Pattern matching scoring
            for pattern in self.plate_patterns:
                if re.match(pattern, text):
                    score *= 1.3
                    break
            
            return min(score, 2.0)  # Cap at 2.0
            
        except Exception as e:
            print(f"Error calculating aggressive quality score: {e}")
            return 1.0
    
    def _draw_aggressive_detection(self, image: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Draw aggressive detection with detailed information
        
        Args:
            image: Original image
            detection: Detection result
            
        Returns:
            np.ndarray: Aggressively annotated image
        """
        try:
            annotated = image.copy()
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['final_confidence']
            method = detection['method']
            text = detection['text']
            
            # Color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green
            elif confidence > 0.5:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw main bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw inner rectangle
            cv2.rectangle(annotated, (x1+2, y1+2), (x2-2, y2-2), (255, 255, 255), 1)
            
            # Draw outer rectangle
            cv2.rectangle(annotated, (x1-2, y1-2), (x2+2, y2+2), color, 1)
            
            # Enhanced label
            label_lines = [
                f"PLATE: {text}",
                f"CONF: {confidence:.3f}",
                f"METHOD: {method}"
            ]
            
            # Draw label background
            label_height = len(label_lines) * 20 + 10
            cv2.rectangle(annotated, (x1, y1 - label_height), 
                          (x1 + 300, y1), color, -1)
            
            # Draw label text
            for i, line in enumerate(label_lines):
                cv2.putText(annotated, line, (x1 + 5, y1 - (label_height - 20) + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return annotated
            
        except Exception as e:
            print(f"Error drawing aggressive detection: {e}")
            return image
    
    def _get_image_hash(self, image: np.ndarray) -> str:
        """Get image hash for duplicate detection"""
        try:
            h, w = image.shape[:2]
            mean_val = np.mean(image)
            return f"{h}_{w}_{mean_val:.0f}"
        except Exception as e:
            print(f"Error getting image hash: {e}")
            return ""
    
    def _is_duplicate(self, image_hash: str) -> bool:
        """Check if image is duplicate within 5-minute window"""
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
        """Cache detection result"""
        try:
            self.detection_cache[image_hash] = {
                'detection': detection,
                'timestamp': datetime.now()
            }
            
            # Clean old cache entries
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
        """Detect license plate from PIL Image"""
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
    """Simple fallback detector for compatibility"""
    
    def __init__(self):
        self.detector = AggressiveLicensePlateDetector()
    
    def detect_license_plate(self, image_path: str) -> Dict:
        """Use aggressive detector"""
        return self.detector.detect_license_plate(image_path)
    
    def detect_license_plate_with_ocr(self, image_path: str, ocr_engine=None) -> Dict:
        """Use aggressive detector"""
        return self.detector.detect_license_plate_with_ocr(image_path, ocr_engine)
    
    def detect_from_pil_image(self, pil_image: Image.Image) -> Dict:
        """Use aggressive detector"""
        return self.detector.detect_from_pil_image(pil_image)
