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

class OptimizedLicensePlateDetector:
    """
    Optimized License Plate Detection with Enhanced Text Preprocessing
    Focused improvements for better capturing and reading accuracy
    """
    
    def __init__(self):
        """Initialize detector with optimized text preprocessing"""
        try:
            # Initialize EasyOCR reader with optimized settings
            self.reader = easyocr.Reader(
                ['en'], 
                gpu=torch.cuda.is_available(),
                verbose=False,
                detector=True,
                recognizer=True
            )
            
            # Optimized detection parameters
            self.confidence_threshold = 0.2  # Balanced threshold
            self.processing_size = (1024, 768)  # Higher resolution for better accuracy
            self.detection_cache = {}
            
            # Enhanced license plate patterns with more specific validation
            self.plate_patterns = [
                r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',  # MH12AB1234
                r'^[A-Z]{2}[0-9]{1}[A-Z]{2}[0-9]{4}$',   # MH1AB1234
                r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$',   # MH12A1234
                r'^[A-Z]{3}[0-9]{3}$',                     # ABC123
                r'^[A-Z]{2}[0-9]{4}$',                     # AB1234
                r'^[A-Z]{3}[0-9]{4}$',                     # ABC1234
                r'^[0-9]{2}[A-Z]{2}[0-9]{4}$',          # 12AB1234
                r'^[A-Z]{1,3}[0-9]{1,4}$',              # A123, AB123, ABC123
                r'^[A-Z0-9]{4,10}$'                      # Optimized alphanumeric
            ]
            
            # Character substitution mappings for better OCR accuracy
            self.char_substitutions = {
                '0': 'O', 'O': '0',
                '1': 'I', 'I': '1', 'L': '1',
                '5': 'S', 'S': '5',
                '6': 'G', 'G': '6',
                '8': 'B', 'B': '8',
                '2': 'Z', 'Z': '2',
                'D': '0', 'Q': '0',
                'T': '1', 'Y': '1'
            }
            
            print("Optimized License Plate Detector initialized successfully")
            print(f"GPU Available: {torch.cuda.is_available()}")
            
        except Exception as e:
            print(f"Error initializing optimized detector: {e}")
            self.reader = None
    
    def detect_license_plate(self, image_path: str) -> Dict:
        """
        Optimized license plate detection with enhanced text preprocessing
        
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
            
            # Optimized preprocessing
            processed_image = self._optimized_preprocessing(image)
            
            # Enhanced detection methods
            all_detections = []
            
            # Method 1: Optimized text detection with enhanced preprocessing
            optimized_detections = self._optimized_text_detection(processed_image)
            for detection in optimized_detections:
                detection['method'] = 'optimized_text'
                detection['weight'] = 1.0
                all_detections.append(detection)
            
            # Method 2: Multi-scale text detection
            scale_detections = self._multi_scale_text_detection(processed_image)
            for detection in scale_detections:
                detection['method'] = 'multi_scale_text'
                detection['weight'] = 0.9
                all_detections.append(detection)
            
            # Method 3: Region-based text detection
            region_detections = self._region_based_text_detection(processed_image)
            for detection in region_detections:
                detection['method'] = 'region_based_text'
                detection['weight'] = 0.8
                all_detections.append(detection)
            
            # Method 4: Enhanced contour detection
            contour_detections = self._enhanced_contour_detection(processed_image)
            for detection in contour_detections:
                detection['method'] = 'enhanced_contour'
                detection['weight'] = 0.7
                all_detections.append(detection)
            
            # Select best result
            best_detection = self._select_best_optimized_detection(all_detections, image)
            
            if best_detection:
                # Cache detection
                self._cache_detection(image_hash, best_detection)
                
                # Draw detection
                annotated_image = self._draw_optimized_detection(image, best_detection)
                
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
            print(f"Error in optimized license plate detection: {e}")
            return {
                "plate_number": "PROCESSING_ERROR", 
                "confidence": 0, 
                "error": str(e)
            }
    
    def detect_license_plate_with_ocr(self, image_path: str, ocr_engine=None) -> Dict:
        """Detection with OCR integration (for compatibility)"""
        return self.detect_license_plate(image_path)
    
    def _optimized_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized image preprocessing for better text extraction
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Optimized preprocessed image
        """
        try:
            # Resize to higher resolution for better accuracy
            h, w = image.shape[:2]
            if h < self.processing_size[1] or w < self.processing_size[0]:
                scale = min(self.processing_size[0]/w, self.processing_size[1]/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Convert to LAB for better color separation
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE with optimized parameters
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter with optimized parameters
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            
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
            print(f"Error in optimized preprocessing: {e}")
            return image
    
    def _optimized_text_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Optimized text detection with enhanced preprocessing
        
        Args:
            image (np.ndarray): Preprocessed image
            
        Returns:
            List[Dict]: Optimized text detections
        """
        try:
            if self.reader is None:
                return []
            
            # Convert back to BGR for EasyOCR
            if len(image.shape) == 2:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image
            
            # Use EasyOCR with optimized settings
            results = self.reader.readtext(image_bgr)
            
            detections = []
            
            for (bbox, text, confidence) in results:
                # Enhanced text cleaning
                cleaned_text = self._enhanced_text_cleaning(text)
                
                if self._is_optimized_plate_text(cleaned_text):
                    # Convert bbox format
                    x1 = int(min([point[0] for point in bbox]))
                    y1 = int(min([point[1] for point in bbox]))
                    x2 = int(max([point[0] for point in bbox]))
                    y2 = int(max([point[1] for point in bbox]))
                    
                    # Optimized bounding box expansion
                    expanded_bbox = self._optimized_bbox_expansion((x1, y1, x2, y2), image.shape)
                    
                    detections.append({
                        'text': cleaned_text,
                        'confidence': confidence,
                        'bbox': expanded_bbox,
                        'raw_text': text
                    })
            
            return detections
            
        except Exception as e:
            print(f"Error in optimized text detection: {e}")
            return []
    
    def _multi_scale_text_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Multi-scale text detection for better accuracy
        
        Args:
            image (np.ndarray): Preprocessed image
            
        Returns:
            List[Dict]: Multi-scale text detections
        """
        try:
            detections = []
            scales = [0.8, 1.0, 1.2, 1.4]
            
            for scale in scales:
                # Resize image
                h, w = image.shape[:2]
                new_h = int(h * scale)
                new_w = int(w * scale)
                scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                if self.reader:
                    # Convert to BGR for EasyOCR
                    if len(scaled_image.shape) == 2:
                        image_bgr = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)
                    else:
                        image_bgr = scaled_image
                    
                    results = self.reader.readtext(image_bgr)
                    
                    for (bbox, text, confidence) in results:
                        cleaned_text = self._enhanced_text_cleaning(text)
                        
                        if self._is_optimized_plate_text(cleaned_text):
                            # Scale coordinates back
                            x1 = int(min([point[0] for point in bbox]) / scale)
                            y1 = int(min([point[1] for point in bbox]) / scale)
                            x2 = int(max([point[0] for point in bbox]) / scale)
                            y2 = int(max([point[1] for point in bbox]) / scale)
                            
                            expanded_bbox = self._optimized_bbox_expansion((x1, y1, x2, y2), image.shape)
                            
                            detections.append({
                                'text': cleaned_text,
                                'confidence': confidence * 0.9,  # Slightly lower confidence for scaled
                                'bbox': expanded_bbox,
                                'raw_text': text,
                                'scale': scale
                            })
            
            return detections
            
        except Exception as e:
            print(f"Error in multi-scale text detection: {e}")
            return []
    
    def _region_based_text_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Region-based text detection for better accuracy
        
        Args:
            image (np.ndarray): Preprocessed image
            
        Returns:
            List[Dict]: Region-based text detections
        """
        try:
            detections = []
            h, w = image.shape[:2]
            
            # Define regions where license plates are likely to be
            regions = [
                (0, int(h * 0.3), w, int(h * 0.9)),  # Lower 70% of image
                (int(w * 0.1), int(h * 0.2), int(w * 0.9), int(h * 0.8)),  # Center region
                (int(w * 0.2), int(h * 0.4), int(w * 0.8), int(h * 0.9)),  # Lower center
            ]
            
            for i, (x1, y1, x2, y2) in enumerate(regions):
                # Extract region
                region = image[y1:y2, x1:x2]
                
                if region.size > 0 and self.reader:
                    # Convert to BGR for EasyOCR
                    if len(region.shape) == 2:
                        region_bgr = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
                    else:
                        region_bgr = region
                    
                    results = self.reader.readtext(region_bgr)
                    
                    for (bbox, text, confidence) in results:
                        cleaned_text = self._enhanced_text_cleaning(text)
                        
                        if self._is_optimized_plate_text(cleaned_text):
                            # Adjust coordinates to full image
                            full_x1 = x1 + int(min([point[0] for point in bbox]))
                            full_y1 = y1 + int(min([point[1] for point in bbox]))
                            full_x2 = x1 + int(max([point[0] for point in bbox]))
                            full_y2 = y1 + int(max([point[1] for point in bbox]))
                            
                            expanded_bbox = self._optimized_bbox_expansion((full_x1, full_y1, full_x2, full_y2), image.shape)
                            
                            detections.append({
                                'text': cleaned_text,
                                'confidence': confidence * 0.8,  # Lower confidence for region-based
                                'bbox': expanded_bbox,
                                'raw_text': text,
                                'region': i
                            })
            
            return detections
            
        except Exception as e:
            print(f"Error in region-based text detection: {e}")
            return []
    
    def _enhanced_contour_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Enhanced contour detection with better validation
        
        Args:
            image (np.ndarray): Preprocessed image
            
        Returns:
            List[Dict]: Enhanced contour detections
        """
        try:
            # Find contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Enhanced validation
                area = cv2.contourArea(contour)
                aspect_ratio = w / h
                image_area = image.shape[0] * image.shape[1]
                area_ratio = area / image_area
                
                # More specific validation for license plates
                if (1.5 <= aspect_ratio <= 6.0 and 
                    1000 <= area <= 50000 and 
                    0.001 <= area_ratio <= 0.1 and
                    w > 60 and h > 15):
                    
                    # Extract region
                    if y + h <= image.shape[0] and x + w <= image.shape[1]:
                        region = image[y:y+h, x:x+w]
                        
                        # Enhanced text validation
                        if self._has_optimized_text_characteristics(region):
                            # Extract text using EasyOCR
                            if self.reader:
                                # Convert to BGR for EasyOCR
                                region_bgr = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
                                
                                results = self.reader.readtext(region_bgr)
                                
                                for (bbox, text, confidence) in results:
                                    cleaned_text = self._enhanced_text_cleaning(text)
                                    
                                    if self._is_optimized_plate_text(cleaned_text):
                                        detections.append({
                                            'text': cleaned_text,
                                            'confidence': confidence * 0.7,  # Lower confidence for contour
                                            'bbox': (x, y, x + w, y + h),
                                            'raw_text': text,
                                            'area': area,
                                            'aspect_ratio': aspect_ratio
                                        })
            
            return detections
            
        except Exception as e:
            print(f"Error in enhanced contour detection: {e}")
            return []
    
    def _enhanced_text_cleaning(self, text: str) -> str:
        """
        Enhanced text cleaning with better character recognition
        
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
            
            # Apply character substitutions based on context
            cleaned = self._apply_contextual_substitutions(cleaned)
            
            return cleaned.strip()
            
        except Exception as e:
            print(f"Error in enhanced text cleaning: {e}")
            return ""
    
    def _apply_contextual_substitutions(self, text: str) -> str:
        """
        Apply contextual character substitutions for better accuracy
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with contextual substitutions
        """
        try:
            if not text:
                return text
            
            # Apply substitutions based on position and context
            result = []
            
            for i, char in enumerate(text):
                # Check if character should be substituted
                if char in self.char_substitutions:
                    substitution = self.char_substitutions[char]
                    
                    # Context-based substitution logic
                    if self._should_substitute_char(text, i, char, substitution):
                        result.append(substitution)
                    else:
                        result.append(char)
                else:
                    result.append(char)
            
            return ''.join(result)
            
        except Exception as e:
            print(f"Error applying contextual substitutions: {e}")
            return text
    
    def _should_substitute_char(self, text: str, position: int, original: str, substitution: str) -> bool:
        """
        Determine if character should be substituted based on context
        
        Args:
            text: Full text
            position: Character position
            original: Original character
            substitution: Substitution character
            
        Returns:
            bool: True if substitution should be applied
        """
        try:
            # Context-aware substitution rules
            if original in ['0', 'O']:
                # In license plates, letters are more common in certain positions
                if position < len(text) // 2:
                    return substitution == 'O'  # Prefer letters in first half
                else:
                    return substitution == '0'  # Prefer numbers in second half
                    
            elif original in ['1', 'I', 'L']:
                # I and L are more common in letter positions
                if position < len(text) // 2:
                    return substitution in ['I', 'L']
                else:
                    return substitution == '1'
                    
            elif original in ['5', 'S']:
                # S is more common in letter positions
                if position < len(text) // 2:
                    return substitution == 'S'
                else:
                    return substitution == '5'
                    
            elif original in ['6', 'G']:
                # G is more common in letter positions
                if position < len(text) // 2:
                    return substitution == 'G'
                else:
                    return substitution == '6'
                    
            elif original in ['8', 'B']:
                # B is more common in letter positions
                if position < len(text) // 2:
                    return substitution == 'B'
                else:
                    return substitution == '8'
                    
            elif original in ['2', 'Z']:
                # Z is more common in letter positions
                if position < len(text) // 2:
                    return substitution == 'Z'
                else:
                    return substitution == '2'
            
            return False
            
        except Exception as e:
            print(f"Error in substitution logic: {e}")
            return False
    
    def _is_optimized_plate_text(self, text: str) -> bool:
        """
        Optimized validation for license plate text
        
        Args:
            text (str): Text to validate
            
        Returns:
            bool: True if text is likely a license plate
        """
        try:
            if not text or len(text) < 3 or len(text) > 12:
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
            
            # Optimized alphanumeric check
            if re.match(r'^[A-Z0-9]{4,10}$', text):
                return True
            
            return False
            
        except Exception as e:
            print(f"Error validating optimized plate text: {e}")
            return False
    
    def _has_optimized_text_characteristics(self, region: np.ndarray) -> bool:
        """
        Optimized validation for text-like characteristics
        
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
            
            # Optimized white ratio range
            if not (0.25 <= white_ratio <= 0.75):
                return False
            
            # Connected components analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # Optimized component count
            if not (4 <= num_labels <= 20):
                return False
            
            # Analyze component sizes
            component_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
            if component_areas:
                avg_area = np.mean(component_areas)
                std_area = np.std(component_areas)
                
                # Check for consistent component sizes
                if std_area > avg_area * 1.5:  # Less variation allowed
                    return False
            
            # Horizontal projection analysis
            h_projection = np.sum(binary, axis=1)
            h_variance = np.var(h_projection)
            
            # Text should have variation in horizontal projection
            if h_variance < 800:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error in optimized text characteristics: {e}")
            return False
    
    def _optimized_bbox_expansion(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Optimized bounding box expansion for better plate region capture
        
        Args:
            bbox: Original bounding box (x1, y1, x2, y2)
            image_shape: Image shape (h, w, c)
            
        Returns:
            Tuple[int, int, int, int]: Optimized expanded bounding box
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = image_shape[:2]
            
            # Calculate expansion based on bbox size
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Optimized expansion based on aspect ratio
            aspect_ratio = bbox_width / bbox_height
            
            if aspect_ratio > 3.0:  # Wide plate - expand more horizontally
                expand_x = max(15, int(bbox_width * 0.2))
                expand_y = max(10, int(bbox_height * 0.15))
            else:  # Standard plate - balanced expansion
                expand_x = max(12, int(bbox_width * 0.18))
                expand_y = max(12, int(bbox_height * 0.18))
            
            # Apply expansion with boundaries
            x1 = max(0, x1 - expand_x)
            y1 = max(0, y1 - expand_y)
            x2 = min(w, x2 + expand_x)
            y2 = min(h, y2 + expand_y)
            
            return (x1, y1, x2, y2)
            
        except Exception as e:
            print(f"Error in optimized bbox expansion: {e}")
            return bbox
    
    def _select_best_optimized_detection(self, detections: List[Dict], image: np.ndarray) -> Optional[Dict]:
        """
        Select best detection from optimized methods
        
        Args:
            detections: List of all detections
            image: Original image
            
        Returns:
            Dict: Best detection or None
        """
        try:
            if not detections:
                return None
            
            # Calculate optimized confidence for each detection
            for detection in detections:
                base_confidence = detection['confidence']
                weight = detection.get('weight', 0.7)
                detection['final_confidence'] = base_confidence * weight
                
                # Quality scoring
                bbox = detection['bbox']
                quality_score = self._calculate_optimized_quality_score(bbox, detection['text'], image, detection.get('method', ''))
                detection['final_confidence'] *= quality_score
            
            # Select best detection
            best_detection = max(detections, key=lambda x: x['final_confidence'])
            
            return best_detection
            
        except Exception as e:
            print(f"Error selecting best optimized detection: {e}")
            return None
    
    def _calculate_optimized_quality_score(self, bbox: Tuple[int, int, int, int], text: str, image: np.ndarray, method: str) -> float:
        """
        Calculate optimized quality score for detection
        
        Args:
            bbox: Bounding box
            text: Detected text
            image: Original image
            method: Detection method
            
        Returns:
            float: Optimized quality score (0.0 to 1.8)
        """
        try:
            score = 1.0
            
            # Method-based scoring
            if 'optimized_text' in method:
                score *= 1.4  # Highest boost for optimized text
            elif 'multi_scale' in method:
                score *= 1.3  # High boost for multi-scale
            elif 'region_based' in method:
                score *= 1.2  # Medium boost for region-based
            elif 'enhanced_contour' in method:
                score *= 1.1  # Small boost for enhanced contour
            
            # Text length scoring
            if 5 <= len(text) <= 8:
                score *= 1.3
            elif 4 <= len(text) <= 10:
                score *= 1.2
            elif 3 <= len(text) <= 11:
                score *= 1.1
            
            # Aspect ratio scoring
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            aspect_ratio = w / h
            
            if 2.0 <= aspect_ratio <= 4.5:
                score *= 1.25
            elif 1.5 <= aspect_ratio <= 6.0:
                score *= 1.15
            
            # Position scoring (plates usually in lower half)
            rel_y = y1 / image.shape[0]
            if 0.3 <= rel_y <= 0.9:
                score *= 1.1
            
            # Pattern matching scoring
            for pattern in self.plate_patterns:
                if re.match(pattern, text):
                    score *= 1.3
                    break
            
            return min(score, 1.8)  # Cap at 1.8
            
        except Exception as e:
            print(f"Error calculating optimized quality score: {e}")
            return 1.0
    
    def _draw_optimized_detection(self, image: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Draw optimized detection with enhanced visualization
        
        Args:
            image: Original image
            detection: Detection result
            
        Returns:
            np.ndarray: Optimized annotated image
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
            
            # Draw corner markers
            corner_length = 15
            corner_thickness = 3
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
            
            # Enhanced label
            label_lines = [
                f"PLATE: {text}",
                f"CONF: {confidence:.3f}",
                f"METHOD: {method}"
            ]
            
            # Draw label background
            label_height = len(label_lines) * 20 + 10
            label_width = max([len(line) * 8 for line in label_lines]) + 20
            cv2.rectangle(annotated, (x1, y1 - label_height), 
                          (x1 + label_width, y1), color, -1)
            
            # Draw label text
            for i, line in enumerate(label_lines):
                cv2.putText(annotated, line, (x1 + 10, y1 - (label_height - 20) + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return annotated
            
        except Exception as e:
            print(f"Error drawing optimized detection: {e}")
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
        self.detector = OptimizedLicensePlateDetector()
    
    def detect_license_plate(self, image_path: str) -> Dict:
        """Use optimized detector"""
        return self.detector.detect_license_plate(image_path)
    
    def detect_license_plate_with_ocr(self, image_path: str, ocr_engine=None) -> Dict:
        """Use optimized detector"""
        return self.detector.detect_license_plate_with_ocr(image_path, ocr_engine)
    
    def detect_from_pil_image(self, pil_image: Image.Image) -> Dict:
        """Use optimized detector"""
        return self.detector.detect_from_pil_image(pil_image)
