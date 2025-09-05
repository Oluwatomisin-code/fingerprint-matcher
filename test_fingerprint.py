#!/usr/bin/env python3
"""
Test script for fingerprint matching functionality
"""

import os
import asyncio
import logging
import time
from typing import List, Dict, Any
from pathlib import Path

from fingerprint_matcher import FingerprintMatcher
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_fingerprint_matching():
    """Test fingerprint matching functionality"""
    logger.info("🚀 Starting Fingerprint Matching Test")
    logger.info("=====================================")

    try:
        # Initialize the fingerprint matcher
        matcher = FingerprintMatcher({
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'aws_region': os.getenv('AWS_REGION'),
            'bucket_name': os.getenv('S3_BUCKET_NAME'),
            'match_threshold': 0.6,  # Lower threshold for testing
            'max_concurrent_downloads': 5,
            'feature_cache_size': 100
        })

        logger.info("✅ Fingerprint matcher initialized")

        # Create a sample fingerprint image path
        sample_image_path = Path(__file__).parent / "sample_fingerprint.jpg"

        # Check if sample image exists, if not create a placeholder
        if not sample_image_path.exists():
            logger.info("⚠️  Sample fingerprint image not found. Creating a placeholder...")
            logger.info("📝 Please replace the sample image with your actual fingerprint image")

            # Create a simple test image using PIL
            from PIL import Image, ImageDraw
            
            # Create a 512x512 grayscale image
            img = Image.new('L', (512, 512), color=128)
            draw = ImageDraw.Draw(img)
            
            # Draw some simple patterns to simulate a fingerprint
            for i in range(0, 512, 20):
                draw.line([(i, 0), (i, 512)], fill=100, width=2)
                draw.line([(0, i), (512, i)], fill=150, width=2)
            
            # Save the placeholder image
            img.save(sample_image_path, 'JPEG', quality=95)
            logger.info("✅ Created placeholder image for testing")

        # Read the sample image
        with open(sample_image_path, 'rb') as f:
            query_image_buffer = f.read()
        
        logger.info(f"📸 Loaded query image: {len(query_image_buffer)} bytes")

        # Test feature extraction
        logger.info("\n🔍 Testing feature extraction...")
        query_features = await matcher.preprocess_and_extract(query_image_buffer)

        if query_features:
            logger.info(f"✅ Extracted {query_features['keypoint_count']} keypoints from query image")
        else:
            logger.info("❌ Failed to extract features from query image")
            return

        # Test S3 connection
        logger.info("\n☁️  Testing S3 connection...")
        try:
            keys = await matcher.get_all_fingerprint_keys()
            logger.info(f"✅ Connected to S3 bucket: {len(keys)} fingerprint images found")

            if len(keys) == 0:
                logger.info("⚠️  No fingerprint images found in S3 bucket")
                logger.info("📝 Please upload some fingerprint images to your S3 bucket for testing")
                return

            # Test matching with a subset of images (for performance)
            test_keys = keys[:10]  # Test with first 10 images
            logger.info(f"🧪 Testing matching with {len(test_keys)} images...")

            start_time = time.time()

            # Simulate the matching process
            matches = []
            for key in test_keys:
                try:
                    features = await matcher.download_and_process_image(key)
                    if features:
                        similarity = matcher.calculate_similarity(query_features, features)

                        if similarity >= matcher.match_threshold:
                            matches.append({
                                'key': key,
                                'similarity': similarity,
                                'keypoint_count': features['keypoint_count']
                            })
                except Exception as error:
                    logger.info(f"⚠️  Error processing {key}: {error}")

            processing_time = (time.time() - start_time) * 1000

            # Sort matches by similarity
            matches.sort(key=lambda x: x['similarity'], reverse=True)

            logger.info(f"\n📊 Test Results:")
            logger.info(f"⏱️  Processing time: {processing_time:.0f}ms")
            logger.info(f"🔍 Images processed: {len(test_keys)}")
            logger.info(f"✅ Matches found: {len(matches)}")
            logger.info(f"🎯 Threshold used: {matcher.match_threshold}")

            if matches:
                logger.info("\n🏆 Top matches:")
                for i, match in enumerate(matches[:5]):
                    logger.info(f"{i + 1}. {match['key']} - Similarity: {match['similarity'] * 100:.2f}%")
            else:
                logger.info("\n❌ No matches found above threshold")

        except Exception as error:
            logger.info(f"❌ S3 connection failed: {error}")
            logger.info("📝 Please check your AWS credentials and S3 bucket configuration")

        # Cleanup
        matcher.cleanup()
        logger.info("\n🧹 Cleanup completed")

    except Exception as error:
        logger.error(f"❌ Test failed: {error}")

async def benchmark_performance():
    """Performance benchmark function"""
    logger.info("\n⚡ Performance Benchmark")
    logger.info("=======================")

    try:
        matcher = FingerprintMatcher({
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'aws_region': os.getenv('AWS_REGION'),
            'bucket_name': os.getenv('S3_BUCKET_NAME'),
            'match_threshold': 0.7,
            'max_concurrent_downloads': 10,
            'feature_cache_size': 500
        })

        # Test different batch sizes
        batch_sizes = [10, 25, 50, 100]
        results = []

        for batch_size in batch_sizes:
            logger.info(f"\n🧪 Testing batch size: {batch_size}")

            start_time = time.time()

            try:
                # Simulate processing with different batch sizes
                keys = await matcher.get_all_fingerprint_keys()
                test_keys = keys[:min(100, len(keys))]

                processed = 0
                for i in range(0, len(test_keys), batch_size):
                    batch = test_keys[i:i + batch_size]
                    tasks = [matcher.download_and_process_image(key) for key in batch]
                    
                    # Process batch concurrently
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    processed += len([r for r in batch_results if r is not None])

                processing_time = (time.time() - start_time) * 1000
                throughput = processed / (processing_time / 1000)

                results.append({
                    'batch_size': batch_size,
                    'processing_time': processing_time,
                    'processed': processed,
                    'throughput': f"{throughput:.2f} images/sec"
                })

                logger.info(f"✅ Processed {processed} images in {processing_time:.0f}ms ({throughput:.2f} images/sec)")

            except Exception as error:
                logger.info(f"❌ Error with batch size {batch_size}: {error}")

        logger.info("\n📊 Benchmark Results:")
        logger.info("Batch Size | Processing Time | Throughput")
        logger.info("-----------|----------------|------------")
        for result in results:
            logger.info(f"{result['batch_size']:10d} | {result['processing_time']:14.0f}ms | {result['throughput']}")

        matcher.cleanup()

    except Exception as error:
        logger.error(f"❌ Benchmark failed: {error}")

async def test_feature_extraction():
    """Test feature extraction functionality"""
    logger.info("\n🔬 Feature Extraction Test")
    logger.info("=========================")

    try:
        matcher = FingerprintMatcher()

        # Create a test image
        from PIL import Image, ImageDraw
        import io

        # Create a test fingerprint image
        img = Image.new('L', (512, 512), color=128)
        draw = ImageDraw.Draw(img)
        
        # Draw ridge patterns
        for i in range(0, 512, 15):
            draw.line([(i, 0), (i, 512)], fill=100, width=3)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        image_bytes = img_buffer.getvalue()

        # Test preprocessing
        logger.info("Testing image preprocessing...")
        processed = await matcher.preprocess_image(image_bytes)
        logger.info(f"✅ Preprocessing completed: {len(processed)} bytes")

        # Test feature extraction
        logger.info("Testing feature extraction...")
        features = await matcher.extract_features(image_bytes)
        
        if features:
            logger.info("✅ Feature extraction successful")
            logger.info(f"   - Perceptual hash: {features['perceptual_hash'][:20]}...")
            logger.info(f"   - Ridge features: {len(features['ridge_features'])}")
            logger.info(f"   - Edge features: {len(features['edge_features'])}")
            logger.info(f"   - Texture features: {len(features['texture_features'])}")
            logger.info(f"   - ORB keypoints: {len(features['orb_keypoints'])}")
            logger.info(f"   - Total keypoints: {features['keypoint_count']}")
        else:
            logger.error("❌ Feature extraction failed")

        # Test similarity calculation
        logger.info("Testing similarity calculation...")
        similarity = matcher.calculate_similarity(features, features)
        logger.info(f"✅ Self-similarity: {similarity * 100:.2f}%")

    except Exception as error:
        logger.error(f"❌ Feature extraction test failed: {error}")

async def test_cache_functionality():
    """Test cache functionality"""
    logger.info("\n💾 Cache Functionality Test")
    logger.info("===========================")

    try:
        matcher = FingerprintMatcher({
            'feature_cache_size': 5  # Small cache for testing
        })

        # Create test images
        from PIL import Image, ImageDraw
        import io

        test_images = []
        for i in range(10):
            img = Image.new('L', (512, 512), color=128 + i * 10)
            draw = ImageDraw.Draw(img)
            draw.rectangle([(100, 100), (400, 400)], fill=200)
            
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=95)
            test_images.append(img_buffer.getvalue())

        logger.info(f"Created {len(test_images)} test images")

        # Test cache behavior
        for i, img_bytes in enumerate(test_images):
            logger.info(f"Processing image {i + 1}...")
            features = await matcher.extract_features(img_bytes)
            
            cache_size = len(matcher.feature_cache)
            logger.info(f"   Cache size: {cache_size}/{matcher.feature_cache_size}")

        logger.info("✅ Cache functionality test completed")

        # Test cache cleanup
        matcher.cleanup()
        logger.info("✅ Cache cleanup completed")

    except Exception as error:
        logger.error(f"❌ Cache test failed: {error}")

async def run_tests():
    """Run all tests"""
    await test_fingerprint_matching()
    await benchmark_performance()
    await test_feature_extraction()
    await test_cache_functionality()

    logger.info("\n🎉 All tests completed!")
    logger.info("\n📝 Next steps:")
    logger.info("1. Upload your fingerprint images to S3")
    logger.info("2. Configure your AWS credentials in .env file")
    logger.info("3. Start the server with: python main.py")
    logger.info("4. Use the API endpoints to match fingerprints")

if __name__ == "__main__":
    asyncio.run(run_tests()) 