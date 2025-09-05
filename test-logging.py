#!/usr/bin/env python3
"""
Simple test script to verify the Python conversion works correctly
"""

import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        import fastapi
        logger.info("✅ FastAPI imported successfully")
    except ImportError as e:
        logger.error(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import cv2
        logger.info("✅ OpenCV imported successfully")
    except ImportError as e:
        logger.error(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        logger.info("✅ NumPy imported successfully")
    except ImportError as e:
        logger.error(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        logger.info("✅ Pillow imported successfully")
    except ImportError as e:
        logger.error(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import boto3
        logger.info("✅ Boto3 imported successfully")
    except ImportError as e:
        logger.error(f"❌ Boto3 import failed: {e}")
        return False
    
    return True

async def test_fingerprint_matcher():
    """Test the fingerprint matcher class"""
    logger.info("Testing fingerprint matcher...")
    
    try:
        from fingerprint_matcher import FingerprintMatcher
        
        # Test initialization
        matcher = FingerprintMatcher({})
        logger.info("✅ FingerprintMatcher initialized successfully")
        
        # Test basic functionality
        from PIL import Image, ImageDraw
        import io
        
        # Create a test image
        img = Image.new('L', (512, 512), color=128)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(100, 100), (400, 400)], fill=200)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        image_bytes = img_buffer.getvalue()
        
        # Test preprocessing
        processed = await matcher.preprocess_image(image_bytes)
        logger.info(f"✅ Image preprocessing successful: {len(processed)} bytes")
        
        # Test feature extraction
        features = await matcher.extract_features(image_bytes)
        if features:
            logger.info(f"✅ Feature extraction successful: {features['keypoint_count']} keypoints")
        else:
            logger.error("❌ Feature extraction failed")
            return False
        
        # Test similarity calculation
        similarity = matcher.calculate_similarity(features, features)
        logger.info(f"✅ Similarity calculation successful: {similarity * 100:.2f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fingerprint matcher test failed: {e}")
        return False

async def test_api_structure():
    """Test that the API structure is correct"""
    logger.info("Testing API structure...")
    
    try:
        import main
        
        # Check that the app exists
        if hasattr(main, 'app'):
            logger.info("✅ FastAPI app found")
        else:
            logger.error("❌ FastAPI app not found")
            return False
        
        # Check that fingerprint_matcher is imported
        if hasattr(main, 'fingerprint_matcher'):
            logger.info("✅ Fingerprint matcher import found")
        else:
            logger.error("❌ Fingerprint matcher import not found")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API structure test failed: {e}")
        return False

async def test_file_structure():
    """Test that all required files exist"""
    logger.info("Testing file structure...")
    
    required_files = [
        'main.py',
        'fingerprint_matcher.py',
        'requirements.txt',
        'test_fingerprint.py',
        'examples/client.py',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            logger.info(f"✅ {file_path} exists")
    
    if missing_files:
        logger.error(f"❌ Missing files: {missing_files}")
        return False
    
    logger.info("✅ All required files exist")
    return True

async def main():
    """Run all tests"""
    logger.info("🚀 Starting Python conversion verification")
    logger.info("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Fingerprint Matcher", test_fingerprint_matcher),
        ("API Structure", test_api_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} test passed")
            else:
                logger.error(f"❌ {test_name} test failed")
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Python conversion successful!")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 