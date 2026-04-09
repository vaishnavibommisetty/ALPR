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
    Robust YOLOv8-based License Plate Detection System
    Optimized for high accuracy and real-world conditions
    """
    
    def __init__(self):
        """Initialize the detector with YOLOv8 model"""
        try:
            # Load YOLOv8 model for license plate detection
            self.model = YOLO('yolov8n.pt')
            
            # Detection parameters
            self.confidence_threshold = 0.5
            self.nms_threshold = 0.4
            self.detection_size = (640, 480)  # Resize for performance
            
            print("YOLOv8 License Plate Detector loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
    
    def detect_license_plate(self, image_path: str) -> Dict:
        """
        Complete license plate detection pipeline
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict: Detection result with plate number and confidence
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {"plate_number": "NOT_DETECTED", "confidence": 0}
            
            # Resize for performance
            resized_image = self._resize_image(image)
            
            # Detect license plates using YOLOv8
            detections = self._detect_plates_yolo(resized_image)
            
            if not detections:
                return {"plate_number": "NOT_DETECTED", "confidence": 0}
            
            # Select best detection (highest confidence)
            best_detection = max(detections, key=lambda x: x['confidence'])
            
            # Scale bounding box back to original image
            scale_x = image.shape[1] / resized_image.shape[1]
            scale_y = image.shape[0] / resized_image.shape[0]
            
            x1 = int(best_detection['bbox'][0] * scale_x)
            y1 = int(best_detection['bbox'][1] * scale_y)
            x2 = int(best_detection['bbox'][2] * scale_x)
            y2 = int(best_detection['bbox'][3] * scale_y)
            
            # Crop plate region
            plate_region = image[y1:y2, x1:x2]
            
            if plate_region.size == 0:
                return {"plate_number": "NOT_DETECTED", "confidence": 0}
            
            # Draw bounding box on original image
            annotated_image = self._draw_bbox(image, (x1, y1, x2, y2), best_detection['confidence'])
            
            # Save annotated image
            output_path = image_path.replace('.', '_detected.')
            cv2.imwrite(output_path, annotated_image)
            
            return {
                "plate_number": "DETECTED_PLATE",  # Will be updated by OCR
                "confidence": best_detection['confidence'],
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
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image for optimal detection performance"""
        try:
            return cv2.resize(image, self.detection_size, interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"Error resizing image: {e}")
            return image
    
    def _detect_plates_yolo(self, image: np.ndarray) -> List[Dict]:
        """
        Detect license plates using YOLOv8
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict]: List of detections with bounding boxes and confidence
        """
        try:
            if self.model is None:
                return []
            
            # Run YOLOv8 detection
            results = self.model(image, conf=self.confidence_threshold, iou=self.nms_threshold)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Filter by confidence
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence
                            })
            
            return detections
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
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
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)  # Green for high confidence, orange for medium
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
            
            # Resize for performance
            resized_image = self._resize_image(image)
            
            # Detect license plates
            detections = self._detect_plates_yolo(resized_image)
            
            if not detections:
                return {"plate_number": "NOT_DETECTED", "confidence": 0}
            
            # Select best detection
            best_detection = max(detections, key=lambda x: x['confidence'])
            
            # Scale bounding box back to original image
            scale_x = image.shape[1] / resized_image.shape[1]
            scale_y = image.shape[0] / resized_image.shape[0]
            
            x1 = int(best_detection['bbox'][0] * scale_x)
            y1 = int(best_detection['bbox'][1] * scale_y)
            x2 = int(best_detection['bbox'][2] * scale_x)
            y2 = int(best_detection['bbox'][3] * scale_y)
            
            # Crop plate region
            plate_region = image[y1:y2, x1:x2]
            
            if plate_region.size == 0:
                return {"plate_number": "NOT_DETECTED", "confidence": 0}
            
            # Draw bounding box
            annotated_image = self._draw_bbox(image, (x1, y1, x2, y2), best_detection['confidence'])
            
            return {
                "plate_number": "DETECTED_PLATE",
                "confidence": best_detection['confidence'],
                "bbox": (x1, y1, x2, y2),
                "plate_image": plate_region,
                "annotated_image": annotated_image
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
                        confidence = min(0.8, aspect_ratio / 4.0)  # Confidence based on aspect ratio
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
