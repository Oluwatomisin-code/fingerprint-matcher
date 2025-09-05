import os
import asyncio
import logging
import hashlib
import base64
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import io

import boto3
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

class FingerprintMatcher:
    """
    High-performance fingerprint matching using OpenCV and AWS S3
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the fingerprint matcher with configuration"""
        config = config or {}
        
        # AWS S3 configuration
        self.aws_access_key_id = config.get('aws_access_key_id') or os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = config.get('aws_secret_access_key') or os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = config.get('aws_region') or os.getenv('AWS_REGION', 'us-east-1')
        self.bucket_name = config.get('bucket_name') or os.getenv('S3_BUCKET_NAME')
        
        # Matching configuration
        self.match_threshold = config.get('match_threshold') or float(os.getenv('MATCH_THRESHOLD', '0.7'))
        self.max_concurrent_downloads = config.get('max_concurrent_downloads') or int(os.getenv('MAX_CONCURRENT_DOWNLOADS', '10'))
        self.feature_cache_size = config.get('feature_cache_size') or int(os.getenv('FEATURE_CACHE_SIZE', '1000'))
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region
        )
        
        # Feature cache
        self.feature_cache = {}
        
        # Advanced fingerprint matching parameters
        self.feature_size = 64
        self.hash_size = 16
        self.edge_threshold = 0.1
        self.ridge_detection_enabled = True
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )
        
        # Initialize FLANN matcher
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        logger.info("Fingerprint matcher initialized successfully")
    
    async def preprocess_image(self, image_buffer: bytes) -> bytes:
        """Preprocess fingerprint image for better feature extraction"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_buffer))
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize to standard size
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Apply sharpening filter
            image = image.filter(ImageFilter.SHARPEN)
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=95)
            return output_buffer.getvalue()
            
        except Exception as error:
            logger.error(f"Error preprocessing image: {error}")
            raise
    
    async def enhance_fingerprint(self, image_buffer: bytes) -> bytes:
        """Enhance fingerprint image for better feature detection"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_buffer))
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize to standard size
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            
            # Enhance brightness
            brightness_enhancer = ImageEnhance.Brightness(image)
            image = brightness_enhancer.enhance(1.1)
            
            # Apply edge enhancement
            image = image.filter(ImageFilter.EDGE_ENHANCE)
            
            # Apply sharpening
            image = image.filter(ImageFilter.SHARPEN)
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=95)
            return output_buffer.getvalue()
            
        except Exception as error:
            logger.error(f"Error enhancing fingerprint: {error}")
            return image_buffer  # Return original if enhancement fails
    
    async def extract_features(self, image_buffer: bytes) -> Dict[str, Any]:
        """Extract advanced features from fingerprint image using multiple techniques"""
        try:
            logger.info("Starting feature extraction...")
            
            # Enhance the image first
            enhanced_buffer = await self.enhance_fingerprint(image_buffer)
            logger.info("Image enhancement completed")
            
            # Create perceptual hash
            perceptual_hash = await self.create_perceptual_hash(enhanced_buffer)
            logger.info(f"Perceptual hash created: {len(perceptual_hash) if perceptual_hash else 0} chars")
            
            # Extract ridge patterns
            ridge_features = await self.extract_ridge_features(enhanced_buffer)
            logger.info(f"Ridge features extracted: {len(ridge_features) if ridge_features else 0} features")
            
            # Extract edge features
            edge_features = await self.extract_edge_features(enhanced_buffer)
            logger.info(f"Edge features extracted: {len(edge_features) if edge_features else 0} features")
            
            # Extract texture features
            texture_features = await self.extract_texture_features(enhanced_buffer)
            logger.info(f"Texture features extracted: {len(texture_features) if texture_features else 0} features")
            
            # Extract ORB features
            orb_features = await self.extract_orb_features(enhanced_buffer)
            logger.info(f"ORB features extracted: {len(orb_features['keypoints']) if orb_features['keypoints'] else 0} keypoints")
            
            # Combine all features with safety checks
            keypoint_count = len(orb_features['keypoints']) if orb_features['keypoints'] else 0
            logger.info(f"Keypoint count: {keypoint_count}")
            
            # Ensure all feature lists are valid
            ridge_features = ridge_features if ridge_features else []
            edge_features = edge_features if edge_features else []
            texture_features = texture_features if texture_features else []
            
            combined_features = {
                'perceptual_hash': perceptual_hash or '',
                'ridge_features': ridge_features,
                'edge_features': edge_features,
                'texture_features': texture_features,
                'orb_keypoints': orb_features['keypoints'],
                'orb_descriptors': orb_features['descriptors'],
                'keypoint_count': keypoint_count
            }
            
            logger.info("Feature extraction completed successfully")
            return combined_features
            
        except Exception as error:
            logger.error(f"Error extracting features: {error}")
            # Return a safe default
            return {
                'perceptual_hash': '',
                'ridge_features': [],
                'edge_features': [],
                'texture_features': [],
                'orb_keypoints': [],
                'orb_descriptors': None,
                'keypoint_count': 0
            }
    
    async def create_perceptual_hash(self, image_buffer: bytes) -> str:
        """Create perceptual hash for image comparison"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_buffer))
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize to 16x16
            image = image.resize((16, 16), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate average pixel value
            average = np.mean(img_array)
            
            # Create hash
            hash_bits = img_array > average
            hash_string = ''.join(['1' if bit else '0' for bit in hash_bits.flatten()])
            
            return hash_string
            
        except Exception as error:
            logger.error(f"Error creating perceptual hash: {error}")
            return ""
    
    async def extract_ridge_features(self, image_buffer: bytes) -> List[float]:
        """Extract ridge patterns from fingerprint"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_buffer))
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convert to numpy array
            img_array = np.array(image)
            
            features = []
            
            # Analyze ridge patterns at different scales
            for scale in [1, 2, 3]:
                try:
                    # Resize image with safety checks
                    new_width = max(1, img_array.shape[1] // scale)
                    new_height = max(1, img_array.shape[0] // scale)
                    scaled_img = cv2.resize(img_array, (new_width, new_height))
                    
                    # Sample ridge patterns
                    for x in range(0, scaled_img.shape[1], 8):
                        for y in range(0, scaled_img.shape[0], 8):
                            if x < scaled_img.shape[1] and y < scaled_img.shape[0]:
                                intensity = float(scaled_img[y, x])
                                if intensity is not None:
                                    features.append(intensity)
                except Exception as e:
                    logger.warning(f"Error processing scale {scale}: {e}")
                    continue
            
            return features[:100]  # Limit to 100 features
            
        except Exception as error:
            logger.error(f"Error extracting ridge features: {error}")
            return []
    
    async def extract_edge_features(self, image_buffer: bytes) -> List[float]:
        """Extract edge features from fingerprint"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_buffer, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # Apply edge detection
            edges = cv2.Canny(img, 50, 150)
            
            features = []
            
            # Sample edge features
            for x in range(0, edges.shape[1], 16):
                for y in range(0, edges.shape[0], 16):
                    if x < edges.shape[1] and y < edges.shape[0]:
                        intensity = float(edges[y, x])
                        if intensity is not None:
                            features.append(intensity)
            
            return features[:50]  # Limit to 50 features
            
        except Exception as error:
            logger.error(f"Error extracting edge features: {error}")
            return []
    
    async def extract_texture_features(self, image_buffer: bytes) -> List[int]:
        """Extract texture features from fingerprint"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_buffer))
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convert to numpy array
            img_array = np.array(image)
            
            features = []
            
            # Calculate local binary patterns
            for x in range(1, img_array.shape[1] - 1, 8):
                for y in range(1, img_array.shape[0] - 1, 8):
                    center_intensity = img_array[y, x]
                    
                    if center_intensity is None:
                        continue
                    
                    pattern = 0
                    neighbors = [
                        img_array[y-1, x-1], img_array[y-1, x], img_array[y-1, x+1],
                        img_array[y, x+1], img_array[y+1, x+1], img_array[y+1, x],
                        img_array[y+1, x-1], img_array[y, x-1]
                    ]
                    
                    for i, neighbor_intensity in enumerate(neighbors):
                        if neighbor_intensity is not None and neighbor_intensity > center_intensity:
                            pattern |= 1 << i
                    
                    features.append(pattern)
            
            return features[:50]  # Limit to 50 features
            
        except Exception as error:
            logger.error(f"Error extracting texture features: {error}")
            return []
    
    async def extract_orb_features(self, image_buffer: bytes) -> Dict[str, Any]:
        """Extract ORB features from fingerprint image"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_buffer, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # Detect ORB keypoints and descriptors
            keypoints, descriptors = self.orb.detectAndCompute(img, None)
            
            return {
                'keypoints': keypoints or [],
                'descriptors': descriptors
            }
            
        except Exception as error:
            logger.error(f"Error extracting ORB features: {error}")
            return {'keypoints': [], 'descriptors': None}
    
    def calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity score between two feature sets"""
        try:
            logger.info("Starting similarity calculation...")
            
            # Ensure we have valid features
            if not features1 or not features2:
                logger.warning("Invalid features provided for similarity calculation")
                return 0.0
            
            logger.info(f"Features1 keys: {list(features1.keys())}")
            logger.info(f"Features2 keys: {list(features2.keys())}")
            
            # Compare perceptual hashes
            logger.info("Calculating hash similarity...")
            hash_similarity = self.calculate_hash_similarity(
                features1.get('perceptual_hash', ''),
                features2.get('perceptual_hash', '')
            )
            
            # Compare ridge features
            logger.info("Calculating ridge similarity...")
            ridge_similarity = self.calculate_feature_similarity(
                features1.get('ridge_features', []),
                features2.get('ridge_features', [])
            )
            
            # Compare edge features
            logger.info("Calculating edge similarity...")
            edge_similarity = self.calculate_feature_similarity(
                features1.get('edge_features', []),
                features2.get('edge_features', [])
            )
            
            # Compare texture features
            logger.info("Calculating texture similarity...")
            texture_similarity = self.calculate_feature_similarity(
                features1.get('texture_features', []),
                features2.get('texture_features', [])
            )
            
            # Compare ORB features
            logger.info("Calculating ORB similarity...")
            orb_similarity = self.calculate_orb_similarity(
                features1.get('orb_descriptors'),
                features2.get('orb_descriptors')
            )
            
            # Weighted combination
            logger.info("Calculating weighted combination...")
            similarity = (
                hash_similarity * 0.2 +
                ridge_similarity * 0.2 +
                edge_similarity * 0.15 +
                texture_similarity * 0.15 +
                orb_similarity * 0.3
            )
            
            # Log detailed similarity scores
            logger.info("🔍 Similarity Analysis:")
            logger.info(f"   📊 Hash Similarity: {hash_similarity * 100:.2f}%")
            logger.info(f"   🏔️  Ridge Similarity: {ridge_similarity * 100:.2f}%")
            logger.info(f"   🔲 Edge Similarity: {edge_similarity * 100:.2f}%")
            logger.info(f"   🧵 Texture Similarity: {texture_similarity * 100:.2f}%")
            logger.info(f"   🎯 ORB Similarity: {orb_similarity * 100:.2f}%")
            logger.info(f"   ⚖️  Weighted Total: {similarity * 100:.2f}%")
            logger.info(f"   🎯 Final Score: {min(similarity, 1.0) * 100:.2f}%")
            
            return min(similarity, 1.0)
            
        except Exception as error:
            logger.error(f"Error calculating similarity: {error}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0
    
    def calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate hash similarity using Hamming distance"""
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 0.0
        
        distance = sum(1 for a, b in zip(hash1, hash2) if a != b)
        return 1.0 - distance / len(hash1)
    
    def calculate_feature_similarity(self, features1: List[float], features2: List[float]) -> float:
        """Calculate feature similarity using correlation"""
        try:
            if not features1 or not features2 or len(features1) == 0 or len(features2) == 0:
                return 0.0
            
            # Filter out None values
            features1 = [f for f in features1 if f is not None]
            features2 = [f for f in features2 if f is not None]
            
            if len(features1) == 0 or len(features2) == 0:
                return 0.0
            
            # Use the shorter array length
            length = min(len(features1), len(features2))
            
            # Calculate correlation coefficient
            sum1 = sum(features1[:length])
            sum2 = sum(features2[:length])
            sum1_sq = sum(x * x for x in features1[:length])
            sum2_sq = sum(x * x for x in features2[:length])
            p_sum = sum(a * b for a, b in zip(features1[:length], features2[:length]))
            
            if length == 0:
                return 0.0
            
            num = p_sum - (sum1 * sum2) / length
            den = ((sum1_sq - (sum1 * sum1) / length) * (sum2_sq - (sum2 * sum2) / length)) ** 0.5
            
            if den == 0:
                return 0.0
            
            return abs(num / den)
        except Exception as error:
            logger.error(f"Error calculating feature similarity: {error}")
            return 0.0
    
    def calculate_orb_similarity(self, descriptors1, descriptors2) -> float:
        """Calculate ORB feature similarity using FLANN matcher"""
        try:
            if descriptors1 is None or descriptors2 is None:
                return 0.0
            
            # Convert descriptors to correct format
            if len(descriptors1) == 0 or len(descriptors2) == 0:
                return 0.0
            
            # ORB descriptors are binary, so we need to use a different approach
            # Convert to uint8 for binary descriptors
            desc1 = np.uint8(descriptors1)
            desc2 = np.uint8(descriptors2)
            
            # Use brute force matcher for binary descriptors instead of FLANN
            bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf_matcher.match(desc1, desc2)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Take top matches (limit to avoid too many matches)
            max_matches = min(50, len(matches))
            good_matches = matches[:max_matches]
            
            # Calculate similarity based on number of good matches
            max_possible_matches = min(len(descriptors1), len(descriptors2))
            if max_possible_matches == 0:
                return 0.0
            
            similarity = len(good_matches) / max_possible_matches
            return min(similarity, 1.0)
            
        except Exception as error:
            logger.error(f"Error calculating ORB similarity: {error}")
            return 0.0
    
    async def get_all_fingerprint_keys(self) -> List[str]:
        """Get all fingerprint images from S3 bucket"""
        try:
            # Check if AWS credentials are configured
            if not self.aws_access_key_id or not self.aws_secret_access_key or not self.bucket_name:
                logger.warning("AWS credentials or bucket name not configured. Returning empty list for testing.")
                return []
            
            keys = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            # Use synchronous pagination since boto3 doesn't support async
            for page in paginator.paginate(Bucket=self.bucket_name):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                            keys.append(key)
            
            return keys
            
        except Exception as error:
            logger.error(f"Error listing S3 objects: {error}")
            logger.warning("Returning empty list due to AWS configuration issues")
            return []
    
    async def download_and_process_image(self, key: str) -> Optional[Dict[str, Any]]:
        """Download and process image from S3"""
        try:
            # Check cache first
            if key in self.feature_cache:
                return self.feature_cache[key]
            
            # Download from S3 (synchronous operation)
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            image_buffer = response['Body'].read()
            
            # Extract features
            features = await self.extract_features(image_buffer)
            
            # Cache the features
            if len(self.feature_cache) >= self.feature_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.feature_cache))
                del self.feature_cache[oldest_key]
            
            self.feature_cache[key] = features
            
            return features
            
        except Exception as error:
            logger.error(f"Error processing image {key}: {error}")
            return None
    
    async def find_matches(self, query_image_buffer: bytes, batch_size: int = 50) -> Dict[str, Any]:
        """Find matching fingerprints efficiently"""
        start_time = datetime.now()
        
        try:
            logger.info("Processing query image...")
            query_features = await self.preprocess_and_extract(query_image_buffer)
            
            if not query_features:
                raise Exception("Failed to extract features from query image")
            
            logger.info("Getting all fingerprint keys from S3...")
            all_keys = await self.get_all_fingerprint_keys()
            logger.info(f"Found {len(all_keys)} fingerprint images in S3")
            
            matches = []
            
            # Process in batches for better performance
            batch_size = batch_size or 50  # Ensure batch_size is not None
            for i in range(0, len(all_keys), batch_size):
                batch = all_keys[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(all_keys) // batch_size) + 1
                logger.info(f"Processing batch {batch_num}/{total_batches}")
                
                # Process batch concurrently
                batch_tasks = []
                for key in batch:
                    task = self.process_single_image(key, query_features)
                    batch_tasks.append(task)
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Filter out None results and exceptions
                for result in batch_results:
                    if isinstance(result, dict) and result is not None:
                        matches.append(result)
            
            # Sort by similarity score (highest first)
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"\n📊 FINAL RESULTS:")
            logger.info(f"⏱️  Processing completed in {processing_time:.0f}ms")
            logger.info(f"🔍 Total images processed: {len(all_keys)}")
            logger.info(f"✅ Matches found: {len(matches)} (above threshold {self.match_threshold * 100:.2f}%)")
            
            if matches:
                logger.info(f"\n🏆 TOP MATCHES:")
                for i, match in enumerate(matches[:5]):
                    logger.info(f"{i + 1}. {match['key']} - Score: {match['similarity'] * 100:.2f}%")
            
            return {
                'matches': matches,
                'total_processed': len(all_keys),
                'processing_time': processing_time,
                'query_keypoint_count': query_features.get('keypoint_count', 0)
            }
            
        except Exception as error:
            logger.error(f"Error in find_matches: {error}")
            raise
    
    async def process_single_image(self, key: str, query_features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single image for matching"""
        try:
            features = await self.download_and_process_image(key)
            if features and features.get('keypoint_count') is not None:
                logger.info(f"\n🔍 Comparing with: {key}")
                similarity = self.calculate_similarity(query_features, features)
                
                if similarity >= self.match_threshold:
                    logger.info(f"✅ MATCH FOUND: {key} - Score: {similarity * 100:.2f}%")
                    return {
                        'key': key,
                        'similarity': similarity,
                        'keypoint_count': features['keypoint_count']
                    }
                else:
                    logger.info(f"❌ No match: {key} - Score: {similarity * 100:.2f}% (below threshold {self.match_threshold * 100:.2f}%)")
            else:
                logger.warning(f"⚠️  Skipping {key}: No valid features extracted")
            
            return None
            
        except Exception as error:
            logger.error(f"Error processing {key}: {error}")
            return None
    
    async def preprocess_and_extract(self, image_buffer: bytes) -> Optional[Dict[str, Any]]:
        """Preprocess and extract features from query image"""
        try:
            processed_buffer = await self.preprocess_image(image_buffer)
            features = await self.extract_features(processed_buffer)
            
            # Ensure we have a valid keypoint_count
            if features and features.get('keypoint_count') is None:
                features['keypoint_count'] = 0
                logger.warning("No keypoints detected, setting keypoint_count to 0")
            
            return features
        except Exception as error:
            logger.error(f"Error preprocessing and extracting features: {error}")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        self.feature_cache.clear()
        logger.info("Feature cache cleared") 