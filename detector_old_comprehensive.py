import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import os
import time
from typing import List, Tuple, Dict, Optional

class LicensePlateDetector:
    """
    Hybrid License Plate Detection System
    Combines YOLOv8 vehicle detection with traditional CV for license plates
    """
    
    def __init__(self):
        """Initialize the detector with YOLOv8 model"""
        try:
            # Load YOLOv8 model for vehicle detection
            self.vehicle_model = YOLO('yolov8n.pt')
            
            # Detection parameters
            self.confidence_threshold = 0.5
            self.nms_threshold = 0.4
            self.detection_size = (640, 480)
            
            # COCO class names for vehicles
            self.vehicle_classes = {
                2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 
                1: 'bicycle', 0: 'person'
            }
            
            print("Hybrid License Plate Detector loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.vehicle_model = None
    
    def detect_license_plate(self, image_path: str) -> Dict:
        """
        Complete license plate detection pipeline using hybrid approach
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict: Detection result with plate number and confidence
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"plate_number": "NOT_DETECTED", "confidence": 0}
            
            # Step 1: Detect vehicles using YOLOv8
            vehicle_detections = self._detect_vehicles(image)
            
            # Step 2: Find license plates using traditional CV
            plate_candidates = self._find_license_plates_traditional(image)
            
            # Step 3: Filter plates within vehicle regions (if vehicles detected)
            if vehicle_detections:
                filtered_plates = self._filter_plates_by_vehicles(plate_candidates, vehicle_detections)
            else:
                # If no vehicles detected, use all plate candidates
                filtered_plates = plate_candidates
            
            # Step 4: Select best plate candidate
            if not filtered_plates:
                return {"plate_number": "NOT_DETECTED", "confidence": 0}
            
            best_plate = max(filtered_plates, key=lambda x: x['confidence'])
            
            # Draw bounding box
            x1, y1, x2, y2 = best_plate['bbox']
            annotated_image = self._draw_bbox(image, (x1, y1, x2, y2), best_plate['confidence'])
            
            # Save annotated image
            output_path = image_path.replace('.', '_detected.')
            cv2.imwrite(output_path, annotated_image)
            
            # Crop plate region
            plate_region = image[y1:y2, x1:x2]
            
            return {
                "plate_number": "DETECTED_PLATE",
                "confidence": best_plate['confidence'],
                "bbox": (x1, y1, x2, y2),
                "plate_image": plate_region,
                "annotated_image_path": output_path
            }
            
        except Exception as e:
            print(f"Error in license plate detection: {e}")
            return {"plate_number": "NOT_DETECTED", "confidence": 0}
    
    def detect_license_plate_with_ocr(self, image_path: str, ocr_engine) -> Dict:
        """
        Complete detection pipeline with OCR
        
        Args:
            image_path (str): Path to the image file
            ocr_engine: OCR engine instance
            
        Returns:
            Dict: Final result with plate number and confidence
        """
        try:
            # Step 1: Detect license plate
            detection_result = self.detect_license_plate(image_path)
            
            if detection_result["plate_number"] == "NOT_DETECTED":
                return {"plate_number": "NOT_DETECTED", "confidence": 0}
            
            # Step 2: Extract text from detected plate
            plate_image = detection_result["plate_image"]
            ocr_result = ocr_engine.extract_text(plate_image)
            
            # Step 3: Validate OCR result
            if ocr_result and ocr_result != "UNKNOWN" and len(ocr_result) >= 4:
                # Combine detection and OCR confidence
                combined_confidence = (detection_result["confidence"] + 0.8) / 2
                
                return {
                    "plate_number": ocr_result,
                    "confidence": combined_confidence,
                    "bbox": detection_result["bbox"],
                    "annotated_image_path": detection_result["annotated_image_path"]
                }
            else:
                return {"plate_number": "NOT_DETECTED", "confidence": 0}
                
        except Exception as e:
            print(f"Error in detection with OCR: {e}")
            return {"plate_number": "NOT_DETECTED", "confidence": 0}
    
    def _detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        """
        Detect vehicles using YOLOv8
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict]: List of vehicle detections
        """
        try:
            if self.vehicle_model is None:
                return []
            
            # Resize for performance
            resized_image = cv2.resize(image, self.detection_size)
            
            # Run YOLOv8 detection
            results = self.vehicle_model(resized_image, conf=self.confidence_threshold, iou=self.nms_threshold)
            
            vehicles = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class and confidence
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Check if it's a vehicle
                        if class_id in self.vehicle_classes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Scale back to original image
                            scale_x = image.shape[1] / resized_image.shape[1]
                            scale_y = image.shape[0] / resized_image.shape[0]
                            
                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y)
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)
                            
                            vehicles.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class': self.vehicle_classes[class_id]
                            })
            
            return vehicles
            
        except Exception as e:
            print(f"Error in vehicle detection: {e}")
            return []
    
    def _find_license_plates_traditional(self, image: np.ndarray) -> List[Dict]:
        """
        Find license plates using traditional computer vision methods
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict]: List of plate candidates
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            candidates = []
            
            # Method 1: Bilateral filter + Canny edge detection
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(bilateral, 30, 200)
            candidates.extend(self._find_plate_contours(edged, gray, "bilateral"))
            
            # Method 2: Gaussian blur + Sobel
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_magnitude = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)
            candidates.extend(self._find_plate_contours(sobel_magnitude, gray, "sobel"))
            
            # Method 3: Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            candidates.extend(self._find_plate_contours(adaptive, gray, "adaptive"))
            
            # Method 4: Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            _, morph = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            candidates.extend(self._find_plate_contours(morph, gray, "morphological"))
            
            return candidates
            
        except Exception as e:
            print(f"Error in traditional plate detection: {e}")
            return []
    
    def _find_plate_contours(self, processed_image: np.ndarray, original_gray: np.ndarray, method: str) -> List[Dict]:
        """
        Find license plate contours in processed image
        
        Args:
            processed_image (np.ndarray): Processed image
            original_gray (np.ndarray): Original grayscale image
            method (str): Detection method name
            
        Returns:
            List[Dict]: List of plate candidates
        """
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
                            'bbox': (x, y, x + w, y + h),
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
        """
        Check if region has characteristics of text/characters
        
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
        """
        Calculate confidence score for plate candidate
        
        Args:
            region (np.ndarray): Plate region
            aspect_ratio (float): Aspect ratio
            area (float): Area in pixels
            
        Returns:
            float: Confidence score
        """
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
    
    def _filter_plates_by_vehicles(self, plates: List[Dict], vehicles: List[Dict]) -> List[Dict]:
        """
        Filter plate candidates that are within vehicle bounding boxes
        
        Args:
            plates (List[Dict]): Plate candidates
            vehicles (List[Dict]): Vehicle detections
            
        Returns:
            List[Dict]: Filtered plate candidates
        """
        try:
            filtered_plates = []
            
            for plate in plates:
                px1, py1, px2, py2 = plate['bbox']
                
                for vehicle in vehicles:
                    vx1, vy1, vx2, vy2 = vehicle['bbox']
                    
                    # Check if plate is within vehicle bounding box (with some tolerance)
                    if (px1 >= vx1 - 20 and py1 >= vy1 - 20 and 
                        px2 <= vx2 + 20 and py2 <= vy2 + 20):
                        filtered_plates.append(plate)
                        break
            
            # If no plates found within vehicles, return all candidates
            if not filtered_plates:
                return plates
            
            return filtered_plates
            
        except Exception as e:
            print(f"Error filtering plates by vehicles: {e}")
            return plates
    
    def _draw_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int], confidence: float) -> np.ndarray:
        """
        Draw bounding box and confidence score on image
        
        Args:
            image (np.ndarray): Input image
            bbox (Tuple): Bounding box coordinates (x1, y1, x2, y2)
            confidence (float): Detection confidence
            
        Returns:
            np.ndarray: Annotated image
        """
        try:
            annotated = image.copy()
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence label
            label = f"License Plate: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                          (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"Error drawing bounding box: {e}")
            return image
    
    def detect_from_pil_image(self, pil_image: Image.Image) -> Dict:
        """
        Detect license plate from PIL Image
        
        Args:
            pil_image (PIL.Image): PIL Image object
            
        Returns:
            Dict: Detection result
        """
        try:
            # Convert PIL to OpenCV format
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Detect vehicles
            vehicle_detections = self._detect_vehicles(image)
            
            # Find license plates
            plate_candidates = self._find_license_plates_traditional(image)
            
            # Filter plates by vehicles
            if vehicle_detections:
                filtered_plates = self._filter_plates_by_vehicles(plate_candidates, vehicle_detections)
            else:
                filtered_plates = plate_candidates
            
            if not filtered_plates:
                return {"plate_number": "NOT_DETECTED", "confidence": 0}
            
            # Select best candidate
            best_plate = max(filtered_plates, key=lambda x: x['confidence'])
            
            return {
                "plate_number": "DETECTED_PLATE",
                "confidence": best_plate['confidence'],
                "bbox": best_plate['bbox']
            }
            
        except Exception as e:
            print(f"Error in PIL image detection: {e}")
            return {"plate_number": "NOT_DETECTED", "confidence": 0}

# Alternative simple detector for fallback
class SimpleLicensePlateDetector:
    """
    Simple fallback detector using traditional computer vision
    """
    
    def __init__(self):
        pass
    
    def detect_license_plate(self, image_path: str) -> Dict:
        """
        Simple license plate detection using OpenCV
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict: Detection result
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {"plate_number": "NOT_DETECTED", "confidence": 0}
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Edge detection
            edged = cv2.Canny(bilateral, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours for license plate candidates
            plate_candidates = []
            
            for contour in contours:
                # Approximate contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
                
                # Check if contour has 4 corners (rectangle)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Check aspect ratio for license plate
                    aspect_ratio = w / h
                    
                    # License plates typically have aspect ratio between 2.0 and 5.5
                    if 2.0 <= aspect_ratio <= 5.5 and w > 80 and h > 20:
                        confidence = min(0.8, aspect_ratio / 4.0)
                        plate_candidates.append({
                            'bbox': (x, y, x + w, y + h),
                            'confidence': confidence
                        })
            
            if plate_candidates:
                # Select best candidate
                best_candidate = max(plate_candidates, key=lambda x: x['confidence'])
                
                # Draw bounding box
                x, y, x2, y2 = best_candidate['bbox']
                cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
                
                # Save annotated image
                output_path = image_path.replace('.', '_detected.')
                cv2.imwrite(output_path, image)
                
                return {
                    "plate_number": "DETECTED_PLATE",
                    "confidence": best_candidate['confidence'],
                    "bbox": best_candidate['bbox'],
                    "annotated_image_path": output_path
                }
            
            return {"plate_number": "NOT_DETECTED", "confidence": 0}
            
        except Exception as e:
            print(f"Error in simple detection: {e}")
            return {"plate_number": "NOT_DETECTED", "confidence": 0}
