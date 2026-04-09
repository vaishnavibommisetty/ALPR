import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import string
from typing import List, Dict, Optional, Tuple

class LicensePlateOCR:
    def __init__(self):
        # Enhanced Tesseract configurations for different scenarios
        self.configs = {
            'single_line': r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'single_block': r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'sparse': r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'dense': r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        }
        
        # Comprehensive license plate patterns for different regions
        self.plate_patterns = {
            'india': [
                r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',  # MH12AB1234
                r'^[A-Z]{2}[0-9]{1,2}[A-Z]{2}[0-9]{4}$',  # MH1AB1234
                r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$',   # MH12A1234
            ],
            'us': [
                r'^[A-Z]{3}[0-9]{3}$',                 # ABC123
                r'^[A-Z]{2}[0-9]{4}$',                 # AB1234
                r'^[A-Z]{3}[0-9]{4}$',                 # ABC1234
            ],
            'uk': [
                r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$',         # AB12CDE
                r'^[A-Z]{3}[0-9]{2}[A-Z]{3}$',         # ABC12DEF
            ],
            'general': [
                r'^[A-Z]{2,3}[0-9]{1,4}[A-Z]{0,3}$',   # General format
                r'^[0-9]{2}[A-Z]{2}[0-9]{4}$',          # 12AB1234
                r'^[A-Z]{1,3}[0-9]{1,4}$',              # A123, AB123, ABC123
            ]
        }
        
        # Common OCR character substitutions
        self.char_substitutions = {
            '0': 'O', 'O': '0',  # Zero/O confusion
            '1': 'I', 'I': '1',  # One/I confusion
            '5': 'S', 'S': '5',  # Five/S confusion
            '6': 'G', 'G': '6',  # Six/G confusion
            '8': 'B', 'B': '8',  # Eight/B confusion
            '2': 'Z', 'Z': '2',  # Two/Z confusion
            'D': '0', 'Q': '0', 'O': '0',  # Letter to number
            'L': '1', 'T': '1', 'I': '1',  # Letter to number
        }
        
        # Plate format validation by length
        self.valid_lengths = [5, 6, 7, 8, 9, 10]
    
    def extract_text(self, plate_image: np.ndarray) -> str:
        """
        Enhanced text extraction with multiple OCR attempts
        
        Args:
            plate_image (numpy.ndarray): Preprocessed license plate image
            
        Returns:
            str: Extracted license plate text
        """
        try:
            # Try multiple OCR configurations
            results = []
            
            # Try multiple OCR configurations and preprocessing methods
            ocr_methods = [
                ('single_line', 'single_line'),
                ('single_block', 'single_block'),
                ('sparse', 'sparse'),
                ('dense', 'dense'),
                ('adaptive', 'single_line'),
                ('adaptive', 'single_block')
            ]
            
            for preprocess_method, config_name in ocr_methods:
                try:
                    # Preprocess specifically for this method
                    processed_image = self._preprocess_for_ocr(plate_image, preprocess_method)
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(processed_image, config=self.configs[config_name])
                    cleaned_text = self._clean_text(text)
                    
                    if cleaned_text and len(cleaned_text) >= 4:
                        results.append({
                            'text': cleaned_text,
                            'config': f"{preprocess_method}_{config_name}",
                            'confidence': self._estimate_confidence(cleaned_text),
                            'method': preprocess_method
                        })
                except Exception as e:
                    continue
            
            # Return the best result
            if results:
                best_result = max(results, key=lambda x: x['confidence'])
                return best_result['text']
            
            return "UNKNOWN"
            
        except Exception as e:
            print(f"Error in enhanced OCR extraction: {e}")
            return "UNKNOWN"
    
    def extract_text_multiple_methods(self, plate_image: np.ndarray) -> str:
        """
        Extract text using multiple OCR methods and return the best result
        
        Args:
            plate_image (numpy.ndarray): License plate image
            
        Returns:
            str: Best extracted license plate text
        """
        results = []
        
        # Method 1: Standard enhanced OCR
        text1 = self.extract_text(plate_image)
        if text1 and text1 != "UNKNOWN":
            results.append({
                'text': text1,
                'method': 'standard',
                'confidence': self._estimate_confidence(text1)
            })
        
        # Method 2: OCR with multiple preprocessing approaches
        preprocessing_methods = ['standard_ocr', 'dense_text', 'sparse_text', 'adaptive_ocr']
        
        for method in preprocessing_methods:
            try:
                processed_image = getattr(self, f'_preprocess_{method}')(plate_image)
                
                # Try multiple Tesseract configurations
                for config_name, config in self.configs.items():
                    try:
                        text = pytesseract.image_to_string(processed_image, config=config)
                        cleaned_text = self._clean_text(text)
                        
                        if cleaned_text and len(cleaned_text) >= 4:
                            results.append({
                                'text': cleaned_text,
                                'method': f'{method}_{config_name}',
                                'confidence': self._estimate_confidence(cleaned_text)
                            })
                    except:
                        continue
            except:
                continue
        
        # Method 3: OCR with inverted image
        try:
            inverted_image = cv2.bitwise_not(plate_image)
            text3 = pytesseract.image_to_string(inverted_image, config=self.configs['single_line'])
            text3 = self._clean_text(text3)
            if text3 and len(text3) >= 5:
                results.append({
                    'text': text3,
                    'method': 'inverted',
                    'confidence': self._estimate_confidence(text3)
                })
        except:
            pass
        
        # Method 4: OCR with threshold variations
        try:
            for thresh in [120, 140, 160, 180]:
                _, thresh_image = cv2.threshold(plate_image, thresh, 255, cv2.THRESH_BINARY)
                text4 = pytesseract.image_to_string(thresh_image, config=self.configs['sparse'])
                text4 = self._clean_text(text4)
                if text4 and len(text4) >= 5:
                    results.append({
                        'text': text4,
                        'method': f'threshold_{thresh}',
                        'confidence': self._estimate_confidence(text4)
                    })
        except:
            pass
        
        # Return the best result
        if results:
            best_result = max(results, key=lambda x: x['confidence'])
            return best_result['text']
        
        return "UNKNOWN"
    
    def _preprocess_for_ocr(self, image):
        """
        Preprocess image for optimal OCR results
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Ensure image is grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            
            # Apply morphological operations to remove noise
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Apply dilation to make characters more prominent
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.dilate(processed, kernel, iterations=1)
            
            return processed
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return image
    
    def _preprocess_for_ocr_v2(self, image):
        """
        Alternative preprocessing method for OCR
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Ensure image is grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Otsu's threshold
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Remove small noise
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            return processed
            
        except Exception as e:
            print(f"Error in preprocessing v2: {e}")
            return image
    
    def _clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning with advanced validation
        
        Args:
            text (str): Raw OCR text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove whitespace and convert to uppercase
        cleaned = text.strip().upper()
        
        # Remove all non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
        
        # Remove extra spaces and line breaks
        cleaned = re.sub(r'\s+', '', cleaned)
        
        # Filter out obviously invalid characters
        cleaned = ''.join(c for c in cleaned if c in string.ascii_uppercase + string.digits)
        
        # Skip if too short or too long
        if len(cleaned) < 4 or len(cleaned) > 12:
            return ""
        
        # Try to fix common OCR errors
        cleaned = self._fix_ocr_errors(cleaned)
        
        # Validate against license plate patterns
        if self._is_valid_plate(cleaned):
            return cleaned
        
        # Try additional cleaning methods
        cleaned = self._advanced_cleaning(cleaned)
        
        # Final validation
        if self._is_valid_plate(cleaned):
            return cleaned
        
        return ""
    
    def _is_valid_plate(self, text: str) -> bool:
        """
        Enhanced validation against multiple license plate patterns
        
        Args:
            text (str): Text to validate
            
        Returns:
            bool: True if valid plate format
        """
        if not text or len(text) < 4 or len(text) > 12:
            return False
        
        # Check against all patterns
        for country, patterns in self.plate_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text):
                    return True
        
        # Additional validation rules
        if self._has_valid_structure(text):
            return True
        
        return False
    
    def _has_valid_structure(self, text: str) -> bool:
        """Check if text has a reasonable license plate structure"""
        # Must contain at least one letter and one number
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        return has_letter and has_number and len(text) in self.valid_lengths
    
    def _fix_ocr_errors(self, text: str) -> str:
        """
        Advanced OCR error correction with context awareness
        
        Args:
            text (str): Text to fix
            
        Returns:
            str: Fixed text
        """
        if not text:
            return ""
        
        # Apply context-aware fixes
        fixed_text = self._context_aware_fix(text)
        
        # Apply pattern-based fixes
        fixed_text = self._pattern_based_fix(fixed_text)
        
        return fixed_text
    
    def _context_aware_fix(self, text: str) -> str:
        """Apply fixes based on character position and context"""
        if len(text) < 4:
            return text
        
        fixed = []
        for i, char in enumerate(text):
            # Determine expected character type based on position
            if i < 2:  # First 2 usually letters
                if char.isdigit():
                    fixed.append(self._char_substitutions.get(char, char))
                else:
                    fixed.append(char)
            elif i >= len(text) - 4:  # Last 4 usually numbers
                if char.isalpha():
                    fixed.append(self._char_substitutions.get(char, char))
                else:
                    fixed.append(char)
            else:  # Middle can be mixed
                fixed.append(char)
        
        return ''.join(fixed)
    
    def _pattern_based_fix(self, text: str) -> str:
        """Apply fixes based on common license plate patterns"""
        # Common patterns and their fixes
        patterns_fixes = [
            (r'([A-Z]{2})(\d{2})([A-Z])(\d{4})', r'\1\2\3\4'),  # Indian format
            (r'([A-Z]{3})(\d{3})', r'\1\2'),                  # US format
            (r'([A-Z]{2})(\d{4})', r'\1\2'),                  # Simple format
        ]
        
        for pattern, replacement in patterns_fixes:
            if re.match(pattern, text):
                return re.sub(pattern, replacement, text)
        
        return text
    
    def _advanced_cleaning(self, text: str) -> str:
        """Advanced cleaning for edge cases"""
        # Remove duplicate characters (common OCR error)
        cleaned = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Fix common character sequences
        cleaned = re.sub(r'0O', 'O0', cleaned)  # Fix 0O sequence
        cleaned = re.sub(r'I1', '1I', cleaned)  # Fix I1 sequence
        
        return cleaned
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence score for extracted text"""
        if not text:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Length bonus
        if 5 <= len(text) <= 10:
            confidence += 0.2
        
        # Pattern match bonus
        if self._is_valid_plate(text):
            confidence += 0.3
        
        # Character diversity bonus
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        if has_letters and has_numbers:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _preprocess_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Adaptive preprocessing for challenging images"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply adaptive thresholding
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 15, 10
            )
            
            # Remove small noise
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
            
            return cleaned
        except Exception as e:
            print(f"Error in adaptive preprocessing: {e}")
            return image
    
    def _preprocess_for_ocr(self, image: np.ndarray, config_name: str = 'single_line') -> np.ndarray:
        """Enhanced OCR preprocessing with multiple techniques"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply different preprocessing based on configuration
            if config_name == 'dense':
                # For dense text - gentle processing
                processed = self._preprocess_dense_text(gray)
            elif config_name == 'sparse':
                # For sparse text - aggressive processing
                processed = self._preprocess_sparse_text(gray)
            elif config_name == 'adaptive':
                # Adaptive preprocessing for challenging images
                processed = self._preprocess_adaptive_ocr(gray)
            else:
                # Standard preprocessing with best practices
                processed = self._preprocess_standard_ocr(gray)
            
            return processed
        except Exception as e:
            print(f"Error in OCR preprocessing: {e}")
            return image
    
    def _preprocess_standard_ocr(self, gray: np.ndarray) -> np.ndarray:
        """Standard OCR preprocessing for license plates"""
        try:
            # Resize for better OCR (increase height)
            h, w = gray.shape
            if h < 50:
                scale_factor = 100 / h
                new_w = int(w * scale_factor)
                gray = cv2.resize(gray, (new_w, 100), interpolation=cv2.INTER_CUBIC)
            
            # Apply bilateral filter to reduce noise while preserving edges
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(bilateral)
            
            # Adaptive thresholding
            adaptive = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
            
            # Final noise removal
            kernel = np.ones((2, 2), np.uint8)
            final = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            return final
            
        except Exception as e:
            print(f"Error in standard OCR preprocessing: {e}")
            return gray
    
    def _preprocess_dense_text(self, gray: np.ndarray) -> np.ndarray:
        """Preprocessing for dense text license plates"""
        try:
            # Gentle blur
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)
            
            # Otsu thresholding
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return binary
            
        except Exception as e:
            print(f"Error in dense text preprocessing: {e}")
            return gray
    
    def _preprocess_sparse_text(self, gray: np.ndarray) -> np.ndarray:
        """Preprocessing for sparse text license plates"""
        try:
            # Resize for better character separation
            h, w = gray.shape
            if h < 60:
                scale_factor = 120 / h
                new_w = int(w * scale_factor)
                gray = cv2.resize(gray, (new_w, 120), interpolation=cv2.INTER_CUBIC)
            
            # Strong bilateral filter
            bilateral = cv2.bilateralFilter(gray, 11, 75, 75)
            
            # High contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(bilateral)
            
            # Aggressive thresholding
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to connect broken characters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return connected
            
        except Exception as e:
            print(f"Error in sparse text preprocessing: {e}")
            return gray
    
    def _preprocess_adaptive_ocr(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive preprocessing for challenging images"""
        try:
            # Multiple preprocessing attempts
            results = []
            
            # Method 1: Standard approach
            results.append(self._preprocess_standard_ocr(gray))
            
            # Method 2: Inverted approach
            inverted = cv2.bitwise_not(gray)
            results.append(self._preprocess_standard_ocr(inverted))
            
            # Method 3: Edge-based approach
            edges = cv2.Canny(gray, 50, 150)
            dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
            results.append(dilated)
            
            # Return the method with best contrast (most white pixels)
            best_result = results[0]
            max_white_pixels = 0
            
            for result in results:
                white_pixels = cv2.countNonZero(result)
                if white_pixels > max_white_pixels:
                    max_white_pixels = white_pixels
                    best_result = result
            
            return best_result
            
        except Exception as e:
            print(f"Error in adaptive OCR preprocessing: {e}")
            return gray
    
    def _select_best_result(self, results):
        """
        Select the best result from multiple OCR attempts
        
        Args:
            results (list): List of OCR results
            
        Returns:
            str: Best result
        """
        if not results:
            return ""
        
        # Score each result based on validity and length
        scored_results = []
        for result in results:
            score = 0
            
            # Check if it matches a valid pattern
            if self._is_valid_plate(result):
                score += 10
            
            # Prefer results with reasonable length (6-10 characters)
            if 6 <= len(result) <= 10:
                score += 5
            
            # Prefer alphanumeric results
            if re.match(r'^[A-Z0-9]+$', result):
                score += 3
            
            scored_results.append((score, result))
        
        # Return the result with the highest score
        if scored_results:
            scored_results.sort(key=lambda x: x[0], reverse=True)
            return scored_results[0][1]
        
        return results[0] if results else ""

# Fallback OCR using simple pattern matching
class SimpleLicensePlateOCR:
    def __init__(self):
        pass
    
    def extract_text(self, plate_image):
        """
        Simple text extraction using template matching (fallback method)
        """
        try:
            # This is a very basic fallback
            # In practice, you'd want to use character templates
            return "UNKNOWN"
        except Exception as e:
            print(f"Error in simple OCR: {e}")
            return ""
