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

class EnhancedLicensePlateDetector:
    """
    Enhanced Contour-Based License Plate Detection
    Optimized for text extraction and reading accuracy
    """
    
    def __init__(self):
        """Initialize detector with enhanced contour detection"""
        try:
            # Initialize EasyOCR reader
            self.reader = easyocr.Reader(
                ['en'], 
                gpu=torch.cuda.is_available(),
                verbose=False
            )
            
            # Enhanced detection parameters
            self.confidence_threshold = 0.3  # Lower threshold for more detection
            self.processing_size = (640, 480)
            self.detection_cache = {}
            
            # Enhanced license plate patterns
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
            
            print("Enhanced Contour-Based License Plate Detector initialized successfully")
            print(f"GPU Available: {torch.cuda.is_available()}")
            
        except Exception as e:
            print(f"Error initializing enhanced detector: {e}")
            self.reader = None
    
    def detect_license_plate(self, image_path: str) -> Dict:
        """
        Enhanced license plate detection with improved contour analysis
        
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
            
            # Check for duplicates
            image_hash = self._get_image_hash(image)
            if self._is_duplicate(image_hash):
                return {"plate_number": "DUPLICATE", "confidence": 0, "error": "Duplicate detection"}
            
            # Enhanced preprocessing
            processed_image = self._enhanced_preprocessing(image)
            
            # Multi-scale contour detection
            all_detections = []
            scales = [0.8, 1.0, 1.2, 1.5]
            
            for scale in scales:
                scale_detections = self._enhanced_contour_detection(processed_image, scale)
                for detection in scale_detections:
                    detection['method'] = f'enhanced_contour_scale_{scale}'
                    detection['weight'] = 0.9  # High confidence for contour method
                    all_detections.append(detection)
            
            # Text-first detection as primary method
            text_detections = self._enhanced_text_detection(processed_image)
            for detection in text_detections:
                detection['method'] = 'enhanced_text_first'
                detection['weight'] = 1.0  # Highest priority
                all_detections.append(detection)
            
            # Select best result
            best_detection = self._select_best_enhanced_detection(all_detections, image)
            
            if best_detection:
                # Cache detection
                self._cache_detection(image_hash, best_detection)
                
                # Draw enhanced bounding box
                annotated_image = self._draw_enhanced_detection(image, best_detection)
                
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
            print(f"Error in enhanced license plate detection: {e}")
            return {
                "plate_number": "PROCESSING_ERROR", 
                "confidence": 0, 
                "error": str(e)
            }
    
    def detect_license_plate_with_ocr(self, image_path: str, ocr_engine=None) -> Dict:
        """Detection with OCR integration (for compatibility)"""
        return self.detect_license_plate(image_path)
    
    def _enhanced_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced image preprocessing for better contour detection
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Enhanced preprocessed image
        """
        try:
            # Convert to LAB color space for better contrast
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter for noise reduction
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Apply adaptive thresholding
            adaptive = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations for text enhancement
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
            
            # Remove small noise
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            return morph
            
        except Exception as e:
            print(f"Error in enhanced preprocessing: {e}")
            return image
    
    def _enhanced_contour_detection(self, image: np.ndarray, scale: float) -> List[Dict]:
        """
        Enhanced contour detection with multiple techniques
        
        Args:
            image (np.ndarray): Preprocessed image
            scale (float): Scale factor
            
        Returns:
            List[Dict]: Enhanced contour detections
        """
        try:
            # Resize image for scale
            h, w = image.shape[:2]
            new_h = int(h * scale)
            new_w = int(w * scale)
            scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            detections = []
            
            # Method 1: Standard contour detection
            contours, _ = cv2.findContours(scaled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_detections = self._analyze_contours(contours, scaled_image, "standard")
            
            # Scale coordinates back
            for detection in contour_detections:
                x1, y1, x2, y2 = detection['bbox']
                detection['bbox'] = (
                    int(x1 / scale), int(y1 / scale), 
                    int(x2 / scale), int(y2 / scale)
                )
                detections.append(detection)
            
            # Method 2: Edge-based contour detection
            edges = cv2.Canny(scaled_image, 50, 150)
            edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edge_detections = self._analyze_contours(edge_contours, scaled_image, "edge")
            
            # Scale coordinates back
            for detection in edge_detections:
                x1, y1, x2, y2 = detection['bbox']
                detection['bbox'] = (
                    int(x1 / scale), int(y1 / scale), 
                    int(x2 / scale), int(y2 / scale)
                )
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error in enhanced contour detection: {e}")
            return []
    
    def _analyze_contours(self, contours: List, image: np.ndarray, method: str) -> List[Dict]:
        """
        Analyze contours for license plate characteristics
        
        Args:
            contours: List of contours
            image: Processed image
            method: Detection method name
            
        Returns:
            List[Dict]: Analyzed contour detections
        """
        try:
            detections = []
            image_area = image.shape[0] * image.shape[1]
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Enhanced validation criteria
                area = cv2.contourArea(contour)
                aspect_ratio = w / h
                area_ratio = area / image_area
                
                # More flexible validation for better detection
                if (1.2 <= aspect_ratio <= 8.0 and  # Wider aspect ratio range
                    500 <= area <= 100000 and      # Larger area range
                    w > 40 and h > 10 and           # Minimum dimensions
                    area_ratio <= 0.15):              # Maximum area ratio
                    
                    # Extract region for OCR validation
                    if y + h <= image.shape[0] and x + w <= image.shape[1]:
                        region = image[y:y+h, x:x+w]
                        
                        # Enhanced text validation
                        if self._has_enhanced_text_characteristics(region):
                            # Extract text using EasyOCR
                            if self.reader:
                                ocr_results = self.reader.readtext(region)
                                
                                for (bbox, text, confidence) in ocr_results:
                                    cleaned_text = self._enhanced_text_cleaning(text)
                                    
                                    if self._is_valid_plate_text(cleaned_text):
                                        detections.append({
                                            'text': cleaned_text,
                                            'confidence': confidence,
                                            'bbox': (x, y, x + w, y + h),
                                            'raw_text': text,
                                            'method': method,
                                            'area': area,
                                            'aspect_ratio': aspect_ratio
                                        })
            
            return detections
            
        except Exception as e:
            print(f"Error analyzing contours: {e}")
            return []
    
    def _enhanced_text_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Enhanced text-first detection with better preprocessing
        
        Args:
            image (np.ndarray): Preprocessed image
            
        Returns:
            List[Dict]: Enhanced text detections
        """
        try:
            if self.reader is None:
                return []
            
            # Convert back to BGR for EasyOCR
            if len(image.shape) == 2:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image
            
            # Use EasyOCR with enhanced parameters
            results = self.reader.readtext(image_bgr)
            
            detections = []
            
            for (bbox, text, confidence) in results:
                # Enhanced text cleaning
                cleaned_text = self._enhanced_text_cleaning(text)
                
                if self._is_valid_plate_text(cleaned_text):
                    # Convert bbox format
                    x1 = int(min([point[0] for point in bbox]))
                    y1 = int(min([point[1] for point in bbox]))
                    x2 = int(max([point[0] for point in bbox]))
                    y2 = int(max([point[1] for point in bbox]))
                    
                    # Enhanced bounding box expansion
                    expanded_bbox = self._enhanced_bbox_expansion((x1, y1, x2, y2), image.shape)
                    
                    detections.append({
                        'text': cleaned_text,
                        'confidence': confidence,
                        'bbox': expanded_bbox,
                        'raw_text': text
                    })
            
            return detections
            
        except Exception as e:
            print(f"Error in enhanced text detection: {e}")
            return []
    
    def _enhanced_text_cleaning(self, text: str) -> str:
        """
        Enhanced text cleaning for better accuracy
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Enhanced cleaned text
        """
        try:
            # Remove non-alphanumeric characters
            cleaned = re.sub(r'[^A-Za-z0-9]', '', text)
            
            # Convert to uppercase
            cleaned = cleaned.upper()
            
            # Remove common OCR errors
            cleaned = re.sub(r'[^\w]', '', cleaned)
            
            # Character substitutions for common OCR mistakes
            substitutions = {
                '0': 'O', 'O': '0',
                '1': 'I', 'I': '1',
                '5': 'S', 'S': '5',
                '6': 'G', 'G': '6',
                '8': 'B', 'B': '8',
                '2': 'Z', 'Z': '2',
            }
            
            # Apply substitutions based on context
            for old_char, new_char in substitutions.items():
                if old_char in cleaned:
                    # Only substitute if it makes sense in license plate context
                    if self._should_substitute(cleaned, old_char, new_char):
                        cleaned = cleaned.replace(old_char, new_char)
            
            return cleaned.strip()
            
        except Exception as e:
            print(f"Error in enhanced text cleaning: {e}")
            return ""
    
    def _should_substitute(self, text: str, old_char: str, new_char: str) -> bool:
        """
        Determine if character substitution should be applied
        
        Args:
            text: Full text
            old_char: Character to replace
            new_char: Replacement character
            
        Returns:
            bool: True if substitution should be applied
        """
        try:
            # Context-aware substitution rules
            if old_char in ['0', 'O']:
                # In license plates, O is more common than 0 in certain positions
                return text.index(old_char) < len(text) // 2
            elif old_char in ['1', 'I']:
                # I is more common than 1 in letter positions
                return text.index(old_char) > len(text) // 3
            elif old_char in ['5', 'S']:
                # S is more common than 5 in letter positions
                return text.index(old_char) > len(text) // 3
            elif old_char in ['6', 'G']:
                # G is more common than 6 in letter positions
                return text.index(old_char) > len(text) // 3
            elif old_char in ['8', 'B']:
                # B is more common than 8 in letter positions
                return text.index(old_char) > len(text) // 3
            elif old_char in ['2', 'Z']:
                # Z is more common than 2 in letter positions
                return text.index(old_char) > len(text) // 3
            
            return False
            
        except Exception as e:
            print(f"Error in substitution logic: {e}")
            return False
    
    def _has_enhanced_text_characteristics(self, region: np.ndarray) -> bool:
        """
        Enhanced validation for text-like characteristics
        
        Args:
            region (np.ndarray): Image region
            
        Returns:
            bool: True if region has text characteristics
        """
        try:
            if region.size == 0:
                return False
            
            # Apply threshold
            _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Count white pixels
            white_pixels = cv2.countNonZero(binary)
            total_pixels = binary.size
            white_ratio = white_pixels / total_pixels
            
            # More flexible white ratio range
            if not (0.2 <= white_ratio <= 0.8):
                return False
            
            # Connected components analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # More flexible component count
            if not (3 <= num_labels <= 25):
                return False
            
            # Analyze component sizes
            component_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
            if component_areas:
                avg_area = np.mean(component_areas)
                std_area = np.std(component_areas)
                
                # Check for consistent component sizes (like characters)
                if std_area > avg_area * 2.0:  # Too much variation
                    return False
            
            # Horizontal projection analysis
            h_projection = np.sum(binary, axis=1)
            h_variance = np.var(h_projection)
            
            # Text should have variation in horizontal projection
            if h_variance < 500:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error in enhanced text characteristics: {e}")
            return False
    
    def _enhanced_bbox_expansion(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Enhanced bounding box expansion for better plate region capture
        
        Args:
            bbox: Original bounding box (x1, y1, x2, y2)
            image_shape: Image shape (h, w, c)
            
        Returns:
            Tuple[int, int, int, int]: Enhanced expanded bounding box
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = image_shape[:2]
            
            # Calculate expansion based on bbox size
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Adaptive expansion (larger for smaller boxes)
            expand_x = max(10, int(bbox_width * 0.15))
            expand_y = max(10, int(bbox_height * 0.15))
            
            # Apply expansion with boundaries
            x1 = max(0, x1 - expand_x)
            y1 = max(0, y1 - expand_y)
            x2 = min(w, x2 + expand_x)
            y2 = min(h, y2 + expand_y)
            
            return (x1, y1, x2, y2)
            
        except Exception as e:
            print(f"Error in enhanced bbox expansion: {e}")
            return bbox
    
    def _select_best_enhanced_detection(self, detections: List[Dict], image: np.ndarray) -> Optional[Dict]:
        """
        Select best detection from enhanced methods
        
        Args:
            detections: List of all detections
            image: Original image
            
        Returns:
            Dict: Best detection or None
        """
        try:
            if not detections:
                return None
            
            # Calculate enhanced confidence for each detection
            for detection in detections:
                base_confidence = detection['confidence']
                weight = detection.get('weight', 0.8)
                detection['final_confidence'] = base_confidence * weight
                
                # Enhanced quality scoring
                bbox = detection['bbox']
                quality_score = self._calculate_enhanced_quality_score(bbox, detection['text'], image, detection.get('method', ''))
                detection['final_confidence'] *= quality_score
            
            # Select best detection
            best_detection = max(detections, key=lambda x: x['final_confidence'])
            
            return best_detection
            
        except Exception as e:
            print(f"Error selecting best enhanced detection: {e}")
            return None
    
    def _calculate_enhanced_quality_score(self, bbox: Tuple[int, int, int, int], text: str, image: np.ndarray, method: str) -> float:
        """
        Calculate enhanced quality score for detection
        
        Args:
            bbox: Bounding box
            text: Detected text
            image: Original image
            method: Detection method
            
        Returns:
            float: Enhanced quality score (0.0 to 1.5)
        """
        try:
            score = 1.0
            
            # Method-based scoring
            if 'text_first' in method:
                score *= 1.3  # Boost for text-first method
            elif 'enhanced_contour' in method:
                score *= 1.2  # Boost for enhanced contour
            elif 'edge' in method:
                score *= 1.1  # Boost for edge method
            
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
                score *= 1.15
            elif 1.5 <= aspect_ratio <= 6.0:
                score *= 1.1
            
            # Position scoring (plates usually in lower half)
            rel_y = y1 / image.shape[0]
            if 0.3 <= rel_y <= 0.9:
                score *= 1.1
            
            # Pattern matching scoring
            for pattern in self.plate_patterns:
                if re.match(pattern, text):
                    score *= 1.2
                    break
            
            return min(score, 1.8)  # Cap at 1.8
            
        except Exception as e:
            print(f"Error calculating enhanced quality score: {e}")
            return 1.0
    
    def _draw_enhanced_detection(self, image: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Draw enhanced detection with detailed information
        
        Args:
            image: Original image
            detection: Detection result
            
        Returns:
            np.ndarray: Enhanced annotated image
        """
        try:
            annotated = image.copy()
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['final_confidence']
            method = detection['method']
            text = detection['text']
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence > 0.6:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw main bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw inner rectangle for better visibility
            cv2.rectangle(annotated, (x1+2, y1+2), (x2-2, y2-2), (255, 255, 255), 1)
            
            # Enhanced label with method info
            label_lines = [
                f"Plate: {text}",
                f"Conf: {confidence:.3f}",
                f"Method: {method}"
            ]
            
            # Draw label background
            label_height = len(label_lines) * 20 + 10
            cv2.rectangle(annotated, (x1, y1 - label_height), 
                          (x1 + 250, y1), color, -1)
            
            # Draw label text
            for i, line in enumerate(label_lines):
                cv2.putText(annotated, line, (x1 + 5, y1 - (label_height - 20) + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return annotated
            
        except Exception as e:
            print(f"Error drawing enhanced detection: {e}")
            return image
    
    def _is_valid_plate_text(self, text: str) -> bool:
        """Enhanced validation for license plate text"""
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
        self.detector = EnhancedLicensePlateDetector()
    
    def detect_license_plate(self, image_path: str) -> Dict:
        """Use enhanced detector"""
        return self.detector.detect_license_plate(image_path)
    
    def detect_license_plate_with_ocr(self, image_path: str, ocr_engine=None) -> Dict:
        """Use enhanced detector"""
        return self.detector.detect_license_plate_with_ocr(image_path, ocr_engine)
    
    def detect_from_pil_image(self, pil_image: Image.Image) -> Dict:
        """Use enhanced detector"""
        return self.detector.detect_from_pil_image(pil_image)
