import cv2
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class FingerprintsMatching:
    """
    Static class for fingerprint matching using minutiae-based approach
    """
    
    @staticmethod
    def fingerprints_matching(image1_path: str, image2_path: str) -> float:
        """
        Match two fingerprint images and return a similarity score
        
        Args:
            image1_path (str): Path to the first fingerprint image
            image2_path (str): Path to the second fingerprint image
            
        Returns:
            float: Similarity score between 0 and 1, where 1 is a perfect match
        """
        try:
            # Load images
            img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                logger.error(f"Failed to load images: {image1_path}, {image2_path}")
                return 0.0
            
            # Preprocess images
            img1 = FingerprintsMatching._preprocess_image(img1)
            img2 = FingerprintsMatching._preprocess_image(img2)
            
            # Extract minutiae features
            minutiae1 = FingerprintsMatching._extract_minutiae(img1)
            minutiae2 = FingerprintsMatching._extract_minutiae(img2)
            
            # Calculate similarity score
            similarity = FingerprintsMatching._calculate_minutiae_similarity(minutiae1, minutiae2)
            
            logger.info(f"Fingerprint matching completed. Similarity score: {similarity}")
            return similarity
            
        except Exception as e:
            logger.error(f"Error in fingerprint matching: {str(e)}")
            return 0.0
    
    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:
        """Preprocess fingerprint image for better minutiae extraction"""
        # Resize to standard size
        image = cv2.resize(image, (512, 512))
        
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Enhance contrast using histogram equalization
        image = cv2.equalizeHist(image)
        
        # Apply morphological operations to enhance ridge structure
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        return image
    
    @staticmethod
    def _extract_minutiae(image: np.ndarray) -> dict:
        """Extract minutiae features from fingerprint image"""
        # Try to apply ridge detection if ximgproc is available
        try:
            ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
            ridges = ridge_filter.getRidgeFilteredImage(image)
        except AttributeError:
            # Fallback: use edge detection instead of ridge detection
            logger.warning("cv2.ximgproc not available, using edge detection instead")
            ridges = cv2.Canny(image, 50, 150)
        
        # Find keypoints using ORB detector
        orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )
        
        keypoints, descriptors = orb.detectAndCompute(image, None)
        
        # Extract additional features
        # Calculate ridge orientation
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        orientation = np.arctan2(sobel_y, sobel_x)
        
        # Calculate ridge frequency
        frequency = FingerprintsMatching._calculate_ridge_frequency(image)
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'orientation': orientation,
            'frequency': frequency,
            'ridges': ridges
        }
    
    @staticmethod
    def _calculate_ridge_frequency(image: np.ndarray) -> float:
        """Calculate average ridge frequency"""
        # Apply FFT to analyze frequency content
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate magnitude spectrum
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Find dominant frequency
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Calculate average frequency in the middle region
        region = magnitude_spectrum[crow-50:crow+50, ccol-50:ccol+50]
        avg_frequency = np.mean(region)
        
        return avg_frequency
    
    @staticmethod
    def _calculate_minutiae_similarity(minutiae1: dict, minutiae2: dict) -> float:
        """Calculate similarity between two sets of minutiae features"""
        try:
            # Compare ORB descriptors
            if minutiae1['descriptors'] is not None and minutiae2['descriptors'] is not None:
                # Use FLANN matcher for descriptor matching
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                   table_number=6,
                                   key_size=12,
                                   multi_probe_level=1)
                search_params = dict(checks=50)
                flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
                
                matches = flann_matcher.knnMatch(minutiae1['descriptors'], minutiae2['descriptors'], k=2)
                
                # Apply ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                descriptor_score = len(good_matches) / max(len(minutiae1['keypoints']), len(minutiae2['keypoints']))
            else:
                descriptor_score = 0.0
            
            # Compare orientation patterns
            orientation_diff = np.mean(np.abs(minutiae1['orientation'] - minutiae2['orientation']))
            orientation_score = max(0, 1 - orientation_diff / np.pi)
            
            # Compare ridge frequency
            freq_diff = abs(minutiae1['frequency'] - minutiae2['frequency'])
            frequency_score = max(0, 1 - freq_diff / max(minutiae1['frequency'], minutiae2['frequency']))
            
            # Calculate weighted similarity score
            weights = [0.6, 0.25, 0.15]  # descriptor, orientation, frequency
            similarity = (weights[0] * descriptor_score + 
                         weights[1] * orientation_score + 
                         weights[2] * frequency_score)
            
            return min(1.0, max(0.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating minutiae similarity: {str(e)}")
            return 0.0 