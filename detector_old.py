import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import os
import time
from typing import List, Tuple, Dict, Optional

class LicensePlateDetector:
    def __init__(self):
        # Load YOLOv8 model for vehicle detection
        try:
            self.vehicle_model = YOLO('yolov8n.pt')
            
            # COCO class names for vehicle detection
            self.vehicle_classes = {
                2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 
                1: 'bicycle', 0: 'person'
            }
            
            # Detection parameters
            self.confidence_threshold = 0.5
            self.nms_threshold = 0.4
            
            print("✓ Enhanced ALPR detector loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO models: {e}")
            self.vehicle_model = None
    
    def detect_license_plate(self, image_path: str) -> Tuple[Optional[np.ndarray], List[np.ndarray], List[List[int]], List[float]]:
        """
        Enhanced license plate detection using traditional computer vision techniques
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (annotated_image, plate_images, bounding_boxes, confidences)
        """
        try:
            # Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return None, [], [], []
            
            # Resize image for performance
            image = self._resize_image(image)
            
            # Detect license plates using traditional computer vision
            plate_candidates = self._detect_plates_traditional(image)
            
            # Process detected plates
            plate_images = []
            bounding_boxes = []
            confidences = []
            
            annotated_image = image.copy()
            
            for i, plate in enumerate(plate_candidates):
                bbox = plate['bbox']
                confidence = plate['confidence']
                
                # Draw bounding box
                self._draw_enhanced_bbox(annotated_image, bbox, confidence, "License Plate")
                
                # Extract and preprocess plate region
                plate_image = self._extract_enhanced_plate(image, bbox)
                plate_images.append(plate_image)
                bounding_boxes.append(bbox)
                confidences.append(confidence)
            
            return annotated_image, plate_images, bounding_boxes, confidences
            
        except Exception as e:
            print(f"Error in license plate detection: {e}")
            return None, [], [], []
    
    def detect_from_pil_image(self, pil_image: Image.Image) -> Tuple[Optional[np.ndarray], List[np.ndarray], List[List[int]], List[float]]:
        """
        Enhanced license plate detection from PIL Image using traditional computer vision
        
        Args:
            pil_image (PIL.Image): PIL Image object
            
        Returns:
            tuple: (annotated_image, plate_images, bounding_boxes, confidences)
        """
        try:
            # Convert PIL to OpenCV format
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Resize if needed
            image = self._resize_image(image)
            
            # Detect license plates using traditional computer vision
            plate_candidates = self._detect_plates_traditional(image)
            
            # Process detected plates
            plate_images = []
            bounding_boxes = []
            confidences = []
            
            annotated_image = image.copy()
            
            for i, plate in enumerate(plate_candidates):
                bbox = plate['bbox']
                confidence = plate['confidence']
                
                # Draw bounding box
                self._draw_enhanced_bbox(annotated_image, bbox, confidence, "License Plate")
                
                # Extract and preprocess plate region
                plate_image = self._extract_enhanced_plate(image, bbox)
                plate_images.append(plate_image)
                bounding_boxes.append(bbox)
                confidences.append(confidence)
            
            return annotated_image, plate_images, bounding_boxes, confidences
            
        except Exception as e:
            print(f"Error in PIL image detection: {e}")
            return None, [], [], []
    
    def _resize_image(self, image: np.ndarray, max_size: int = 1280) -> np.ndarray:
        """Resize image to optimize processing speed"""
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image
    
    def _preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Advanced preprocessing for better object detection"""
        # Convert to LAB color space for better contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _validate_plate_location(self, plate_bbox: List[float], vehicles: List[Dict]) -> bool:
        """Validate if license plate is within vehicle bounding box"""
        px1, py1, px2, py2 = plate_bbox
        plate_center = [(px1 + px2) / 2, (py1 + py2) / 2]
        
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle['bbox']
            # Check if plate center is within vehicle bounds
            if (vx1 <= plate_center[0] <= vx2 and vy1 <= plate_center[1] <= vy2):
                return True
        return False
    
    def _draw_enhanced_bbox(self, image: np.ndarray, bbox: List[float], confidence: float, label: str) -> None:
        """Draw enhanced bounding box with confidence score"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Choose color based on confidence and type
        if 'LICENSE PLATE' in label:
            color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)  # Green or Orange
            thickness = 3
        else:
            color = (255, 0, 255)  # Magenta for vehicles
            thickness = 2
        
        # Draw rounded rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label_text = f"{label} {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Label background
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # Label text
        cv2.putText(image, label_text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _extract_enhanced_plate(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Extract and enhance license plate region"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding around the plate
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        # Extract plate region
        plate_roi = image[y1:y2, x1:x2]
        
        # Apply advanced preprocessing for OCR
        return self._preprocess_plate_image_advanced(plate_roi)
    
    def _preprocess_plate_image_advanced(self, plate_image: np.ndarray) -> np.ndarray:
        """Advanced preprocessing for license plate OCR"""
        try:
            # Convert to grayscale
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image.copy()
            
            # Resize to standard size for consistency
            gray = cv2.resize(gray, (300, 100), interpolation=cv2.INTER_CUBIC)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            # Apply bilateral filter for final smoothing
            final = cv2.bilateralFilter(morph, 9, 75, 75)
            
            return final
            
        except Exception as e:
            print(f"Error in advanced plate preprocessing: {e}")
            return plate_image

    def _detect_plates_traditional(self, image: np.ndarray) -> List[Dict]:
        """Advanced multi-scale license plate detection with ML-based filtering"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale detection
            candidates = []
            scales = [0.8, 1.0, 1.2, 1.5]
            
            for scale in scales:
                scaled_candidates = self._detect_at_scale(gray, scale)
                candidates.extend(scaled_candidates)
            
            # Advanced filtering with ML-based scoring
            filtered_candidates = self._advanced_plate_filtering(candidates, gray, image.shape)
            
            return filtered_candidates
            
        except Exception as e:
            print(f"Error in advanced plate detection: {e}")
            return []
    
    def _detect_at_scale(self, gray: np.ndarray, scale: float) -> List[Dict]:
        """Detect plates at different scales"""
        try:
            # Resize image for scale
            h, w = gray.shape
            new_h = int(h * scale)
            new_w = int(w * scale)
            scaled_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            candidates = []
            
            # Method 1: Enhanced bilateral filter + Multi-threshold Canny
            bilateral = cv2.bilateralFilter(scaled_gray, 9, 20, 20)
            
            # Multi-threshold edge detection
            thresholds = [(30, 200), (50, 150), (70, 200)]
            for low, high in thresholds:
                edged = cv2.Canny(bilateral, low, high)
                scaled_candidates = self._find_plate_contours(edged, scaled_gray, method=f"bilateral_{scale}_{low}_{high}")
                # Scale coordinates back
                for candidate in scaled_candidates:
                    bbox = candidate['bbox']
                    candidate['bbox'] = [
                        int(bbox[0] / scale),
                        int(bbox[1] / scale),
                        int(bbox[2] / scale),
                        int(bbox[3] / scale)
                    ]
                candidates.extend(scaled_candidates)
            
            # Method 2: Enhanced Sobel with direction
            blur = cv2.GaussianBlur(scaled_gray, (3, 3), 0)
            sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_magnitude = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)
            
            scaled_candidates = self._find_plate_contours(sobel_magnitude, scaled_gray, method=f"sobel_{scale}")
            # Scale coordinates back
            for candidate in scaled_candidates:
                bbox = candidate['bbox']
                candidate['bbox'] = [
                    int(bbox[0] / scale),
                    int(bbox[1] / scale),
                    int(bbox[2] / scale),
                    int(bbox[3] / scale)
                ]
            candidates.extend(scaled_candidates)
            
            # Method 3: Advanced adaptive threshold
            adaptive = cv2.adaptiveThreshold(scaled_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            scaled_candidates = self._find_plate_contours(adaptive, scaled_gray, method=f"adaptive_{scale}")
            # Scale coordinates back
            for candidate in scaled_candidates:
                bbox = candidate['bbox']
                candidate['bbox'] = [
                    int(bbox[0] / scale),
                    int(bbox[1] / scale),
                    int(bbox[2] / scale),
                    int(bbox[3] / scale)
                ]
            candidates.extend(scaled_candidates)
            
            return candidates
            
        except Exception as e:
            print(f"Error in scale detection: {e}")
            return []
    
    def _advanced_plate_filtering(self, candidates: List[Dict], gray: np.ndarray, image_shape: tuple) -> List[Dict]:
        """Advanced filtering with ML-based scoring and validation"""
        try:
            if not candidates:
                return []
            
            # Score each candidate with advanced features
            scored_candidates = []
            
            for candidate in candidates:
                bbox = candidate['bbox']
                x, y, x2, y2 = bbox
                w, h = x2 - x, y2 - y
                
                # Extract region
                if x >= 0 and y >= 0 and x2 < gray.shape[1] and y2 < gray.shape[0]:
                    region = gray[y:y2, x:x2]
                    
                    # Calculate advanced features
                    features = self._extract_plate_features(region, bbox, gray)
                    
                    # ML-based scoring
                    ml_score = self._ml_plate_score(features)
                    
                    # Traditional scoring
                    traditional_score = candidate.get('confidence', 0.5)
                    
                    # Combined score
                    combined_score = 0.7 * ml_score + 0.3 * traditional_score
                    
                    candidate['ml_score'] = ml_score
                    candidate['combined_score'] = combined_score
                    candidate['features'] = features
                    
                    scored_candidates.append(candidate)
            
            # Non-maximum suppression with adaptive threshold
            filtered_candidates = self._adaptive_nms(scored_candidates, image_shape)
            
            # Final validation with ensemble methods
            validated_candidates = self._ensemble_validation(filtered_candidates, gray)
            
            return validated_candidates
            
        except Exception as e:
            print(f"Error in advanced filtering: {e}")
            return candidates[:3]  # Fallback to top 3
    
    def _extract_plate_features(self, region: np.ndarray, bbox: List[int], gray: np.ndarray) -> Dict:
        """Extract advanced features for ML-based scoring"""
        try:
            features = {}
            
            # Basic geometric features
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            features['aspect_ratio'] = w / h
            features['area'] = w * h
            features['area_ratio'] = (w * h) / (gray.shape[0] * gray.shape[1])
            
            # Position features
            features['relative_x'] = x / gray.shape[1]
            features['relative_y'] = y / gray.shape[0]
            
            # Texture features
            if region.size > 0:
                # Edge density
                edges = cv2.Canny(region, 50, 150)
                features['edge_density'] = cv2.countNonZero(edges) / edges.size
                
                # Gradient magnitude
                grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                features['gradient_mean'] = np.mean(grad_magnitude)
                features['gradient_std'] = np.std(grad_magnitude)
                
                # Intensity statistics
                features['intensity_mean'] = np.mean(region)
                features['intensity_std'] = np.std(region)
                features['intensity_range'] = np.max(region) - np.min(region)
                
                # Text-like features
                _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                white_ratio = cv2.countNonZero(binary) / binary.size
                features['white_ratio'] = white_ratio
                
                # Connected components
                num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                features['num_components'] = num_labels
                
                # Horizontal projection profile
                h_projection = np.sum(binary, axis=1)
                features['h_projection_variance'] = np.var(h_projection)
                
                # Vertical projection profile
                v_projection = np.sum(binary, axis=0)
                features['v_projection_variance'] = np.var(v_projection)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {}
    
    def _ml_plate_score(self, features: Dict) -> float:
        """Machine learning-based plate scoring"""
        try:
            score = 0.5  # Base score
            
            # Aspect ratio scoring
            if 'aspect_ratio' in features:
                ar = features['aspect_ratio']
                if 2.0 <= ar <= 5.0:
                    score += 0.2
                elif 1.5 <= ar <= 6.0:
                    score += 0.1
            
            # Area scoring
            if 'area_ratio' in features:
                area_ratio = features['area_ratio']
                if 0.001 <= area_ratio <= 0.05:
                    score += 0.15
                elif 0.0005 <= area_ratio <= 0.1:
                    score += 0.05
            
            # Edge density scoring
            if 'edge_density' in features:
                edge_density = features['edge_density']
                if 0.1 <= edge_density <= 0.4:
                    score += 0.15
                elif 0.05 <= edge_density <= 0.5:
                    score += 0.05
            
            # White ratio scoring (text density)
            if 'white_ratio' in features:
                white_ratio = features['white_ratio']
                if 0.3 <= white_ratio <= 0.7:
                    score += 0.1
                elif 0.2 <= white_ratio <= 0.8:
                    score += 0.05
            
            # Component count scoring
            if 'num_components' in features:
                num_components = features['num_components']
                if 5 <= num_components <= 15:
                    score += 0.1
                elif 3 <= num_components <= 20:
                    score += 0.05
            
            # Gradient variance scoring
            if 'gradient_std' in features:
                grad_std = features['gradient_std']
                if 20 <= grad_std <= 80:
                    score += 0.05
            
            # Position scoring (plates usually in lower half)
            if 'relative_y' in features:
                rel_y = features['relative_y']
                if 0.4 <= rel_y <= 0.9:
                    score += 0.05
            
            return min(score, 0.95)
            
        except Exception as e:
            print(f"Error in ML scoring: {e}")
            return 0.5
    
    def _adaptive_nms(self, candidates: List[Dict], image_shape: tuple) -> List[Dict]:
        """Adaptive non-maximum suppression"""
        try:
            if not candidates:
                return []
            
            # Sort by combined score
            candidates.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            
            filtered = []
            image_area = image_shape[0] * image_shape[1]
            
            for candidate in candidates:
                bbox = candidate['bbox']
                score = candidate.get('combined_score', 0)
                
                # Adaptive IoU threshold based on score
                iou_threshold = 0.3 if score > 0.8 else 0.5
                
                # Check for overlap with existing candidates
                is_duplicate = False
                for existing in filtered:
                    existing_bbox = existing['bbox']
                    existing_score = existing.get('combined_score', 0)
                    
                    # Calculate IoU
                    iou = self._calculate_iou(bbox, existing_bbox)
                    
                    # If overlap and current score is not significantly better, skip
                    if iou > iou_threshold and score <= existing_score * 1.1:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered.append(candidate)
            
            return filtered[:5]  # Return top 5 candidates
            
        except Exception as e:
            print(f"Error in adaptive NMS: {e}")
            return candidates[:3]
    
    def _ensemble_validation(self, candidates: List[Dict], gray: np.ndarray) -> List[Dict]:
        """Ensemble validation using multiple methods"""
        try:
            validated = []
            
            for candidate in candidates:
                bbox = candidate['bbox']
                x, y, x2, y2 = bbox
                
                # Extract region
                if x >= 0 and y >= 0 and x2 < gray.shape[1] and y2 < gray.shape[0]:
                    region = gray[y:y2, x:x2]
                    
                    # Multiple validation methods
                    validation_scores = []
                    
                    # Method 1: Text characteristics
                    if self._has_text_characteristics(region):
                        validation_scores.append(0.8)
                    else:
                        validation_scores.append(0.3)
                    
                    # Method 2: Edge-based validation
                    if self._edge_based_validation(region):
                        validation_scores.append(0.7)
                    else:
                        validation_scores.append(0.4)
                    
                    # Method 3: Projection profile validation
                    if self._projection_profile_validation(region):
                        validation_scores.append(0.6)
                    else:
                        validation_scores.append(0.4)
                    
                    # Method 4: Structural validation
                    if self._structural_validation(region, bbox):
                        validation_scores.append(0.7)
                    else:
                        validation_scores.append(0.3)
                    
                    # Average validation score
                    avg_validation = np.mean(validation_scores)
                    
                    # Update confidence with validation
                    candidate['validation_score'] = avg_validation
                    candidate['final_confidence'] = (
                        0.6 * candidate.get('combined_score', 0.5) + 
                        0.4 * avg_validation
                    )
                    
                    # Keep if validation passes threshold
                    if avg_validation > 0.5:
                        validated.append(candidate)
            
            return validated
            
        except Exception as e:
            print(f"Error in ensemble validation: {e}")
            return candidates
    
    def _edge_based_validation(self, region: np.ndarray) -> bool:
        """Validate region based on edge characteristics"""
        try:
            if region.size == 0:
                return False
            
            # Multi-scale edge detection
            edges_50_150 = cv2.Canny(region, 50, 150)
            edges_30_100 = cv2.Canny(region, 30, 100)
            edges_70_200 = cv2.Canny(region, 70, 200)
            
            # Edge density at different scales
            density_50_150 = cv2.countNonZero(edges_50_150) / edges_50_150.size
            density_30_100 = cv2.countNonZero(edges_30_100) / edges_30_100.size
            density_70_200 = cv2.countNonZero(edges_70_200) / edges_70_200.size
            
            # Check if edge density is in reasonable range
            avg_density = (density_50_150 + density_30_100 + density_70_200) / 3
            
            return 0.05 <= avg_density <= 0.3
            
        except Exception as e:
            print(f"Error in edge validation: {e}")
            return False
    
    def _projection_profile_validation(self, region: np.ndarray) -> bool:
        """Validate using projection profile analysis"""
        try:
            if region.size == 0:
                return False
            
            # Threshold the region
            _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Horizontal projection
            h_proj = np.sum(binary, axis=1)
            h_variance = np.var(h_proj)
            
            # Vertical projection
            v_proj = np.sum(binary, axis=0)
            v_variance = np.var(v_proj)
            
            # Text-like regions have characteristic projection patterns
            return h_variance > 1000 and v_variance > 500
            
        except Exception as e:
            print(f"Error in projection validation: {e}")
            return False
    
    def _structural_validation(self, region: np.ndarray, bbox: List[int]) -> bool:
        """Validate based on structural properties"""
        try:
            if region.size == 0:
                return False
            
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            
            # Check minimum size
            if w < 50 or h < 15:
                return False
            
            # Check aspect ratio
            aspect_ratio = w / h
            if aspect_ratio < 1.2 or aspect_ratio > 8.0:
                return False
            
            # Check solidity (ratio of area to bounding box area)
            _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                bbox_area = w * h
                solidity = contour_area / bbox_area
                
                # License plates typically have good solidity
                return solidity > 0.7
            
            return False
            
        except Exception as e:
            print(f"Error in structural validation: {e}")
            return False
    
    def _find_plate_contours(self, processed_image: np.ndarray, original_gray: np.ndarray, method: str = "") -> List[Dict]:
        """Find license plate contours in processed image"""
        try:
            # Find contours
            contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            plate_candidates = []
            
            for contour in contours:
                # Calculate area
                area = cv2.contourArea(contour)
                
                # Filter by area (license plates are typically 1000-50000 pixels)
                if area < 1000 or area > 50000:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio for license plate
                aspect_ratio = w / h
                
                # License plates typically have aspect ratio between 1.5 and 6.0
                if 1.5 <= aspect_ratio <= 6.0 and w > 60 and h > 15:
                    
                    # Additional validation: check if region has text-like characteristics
                    plate_region = original_gray[y:y+h, x:x+w]
                    if self._has_text_characteristics(plate_region):
                        confidence = self._calculate_plate_confidence(plate_region, aspect_ratio, area)
                        
                        plate_candidates.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': confidence,
                            'method': method,
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
            
            return plate_candidates
            
        except Exception as e:
            print(f"Error finding plate contours: {e}")
            return []
    
    def _has_text_characteristics(self, region: np.ndarray) -> bool:
        """Check if region has characteristics of text/characters"""
        try:
            if region.size == 0:
                return False
            
            # Apply threshold
            _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Count white pixels (potential characters)
            white_pixels = cv2.countNonZero(binary)
            total_pixels = binary.size
            white_ratio = white_pixels / total_pixels
            
            # Text regions typically have 30-70% white pixels
            if 0.3 <= white_ratio <= 0.7:
                # Check for connected components (characters)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                
                # License plates typically have 5-15 connected components (characters)
                if 5 <= num_labels <= 15:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking text characteristics: {e}")
            return False
    
    def _calculate_plate_confidence(self, region: np.ndarray, aspect_ratio: float, area: float) -> float:
        """Calculate confidence score for plate candidate"""
        try:
            confidence = 0.5  # Base confidence
            
            # Aspect ratio scoring
            if 2.0 <= aspect_ratio <= 4.0:
                confidence += 0.2
            elif 1.5 <= aspect_ratio <= 5.0:
                confidence += 0.1
            
            # Area scoring
            if 2000 <= area <= 20000:
                confidence += 0.2
            elif 1000 <= area <= 50000:
                confidence += 0.1
            
            # Edge density scoring
            edges = cv2.Canny(region, 50, 150)
            edge_density = cv2.countNonZero(edges) / edges.size
            if 0.1 <= edge_density <= 0.4:
                confidence += 0.1
            
            return min(confidence, 0.95)
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5
    
    def _filter_plate_candidates(self, candidates: List[Dict], image_shape: tuple) -> List[Dict]:
        """Filter and rank plate candidates"""
        try:
            # Remove duplicates (overlapping bounding boxes)
            filtered = []
            
            for candidate in sorted(candidates, key=lambda x: x['confidence'], reverse=True):
                bbox = candidate['bbox']
                
                # Check for overlap with existing candidates
                is_duplicate = False
                for existing in filtered:
                    existing_bbox = existing['bbox']
                    
                    # Calculate IoU (Intersection over Union)
                    iou = self._calculate_iou(bbox, existing_bbox)
                    if iou > 0.3:  # 30% overlap threshold
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered.append(candidate)
            
            # Return top candidates (max 5)
            return filtered[:5]
            
        except Exception as e:
            print(f"Error filtering candidates: {e}")
            return candidates[:3]
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            
            # Calculate union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"Error calculating IoU: {e}")
            return 0.0

# Alternative simple detector using OpenCV (fallback)
class SimpleLicensePlateDetector:
    def __init__(self):
        pass
    
    def detect_license_plate(self, image_path):
        """
        Simple license plate detection using OpenCV contour detection
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None, [], [], []
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Apply edge detection
            edged = cv2.Canny(filtered, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            plate_contour = None
            plate_images = []
            bounding_boxes = []
            confidences = []
            
            for contour in contours:
                # Approximate contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
                
                # If contour has 4 vertices, it might be a license plate
                if len(approx) == 4:
                    plate_contour = approx
                    break
            
            if plate_contour is not None:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(plate_contour)
                
                # Extract license plate region
                plate_roi = image[y:y+h, x:x+w]
                
                # Draw rectangle on original image
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                plate_images.append(plate_roi)
                bounding_boxes.append([x, y, x+w, y+h])
                confidences.append(0.8)  # Fixed confidence for simple detector
            
            return image, plate_images, bounding_boxes, confidences
            
        except Exception as e:
            print(f"Error in simple license plate detection: {e}")
            return None, [], [], []
