#!/usr/bin/env python3
"""
Example script demonstrating the minutiae-based fingerprint matching functionality.
This script shows both direct usage of the FingerprintsMatching class and how to use the API endpoint.
"""

import requests
import os
from fingerprints_matching import FingerprintsMatching

def example_direct_usage():
    """Example of using FingerprintsMatching class directly"""
    print("=== Direct Usage Example ===")
    
    # Example file paths (replace with actual fingerprint images)
    image1_path = "fingerprint1.png"
    image2_path = "fingerprint2.png"
    
    # Check if files exist
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print(f"Please ensure {image1_path} and {image2_path} exist in the current directory")
        print("You can use any fingerprint images for testing")
        return
    
    # Perform matching
    match_score = FingerprintsMatching.fingerprints_matching(image1_path, image2_path)
    
    print(f"Match score: {match_score}")
    
    # Interpret the result
    if match_score >= 0.8:
        print("Result: Excellent match - Very likely same fingerprint")
    elif match_score >= 0.6:
        print("Result: Good match - Likely same fingerprint")
    elif match_score >= 0.4:
        print("Result: Fair match - Possible same fingerprint")
    else:
        print("Result: Poor match - Unlikely same fingerprint")

def example_api_usage():
    """Example of using the /minutiaematch API endpoint"""
    print("\n=== API Usage Example ===")
    
    # API endpoint URL (adjust if running on different port)
    api_url = "http://localhost:8000/minutiaematch"
    
    # Example file path
    fingerprint_path = "fingerprint.png"
    
    # Check if file exists
    if not os.path.exists(fingerprint_path):
        print(f"Please ensure {fingerprint_path} exists in the current directory")
        return
    
    try:
        # Prepare file for upload
        with open(fingerprint_path, 'rb') as f:
            files = {
                'fingerprint': (os.path.basename(fingerprint_path), f, 'image/png')
            }
            data = {
                'threshold': 0.7,
                'max_results': 5
            }
            
            # Make API request
            response = requests.post(api_url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"API Response: {result}")
                
                matches = result['results']['matches']
                print(f"Found {len(matches)} matches")
                
                for i, match in enumerate(matches, 1):
                    print(f"Match {i}: {match['filename']} - Score: {match['similarity']:.3f}")
                    quality = match['match_quality']
                    if quality['excellent_match']:
                        print("  Quality: Excellent match")
                    elif quality['good_match']:
                        print("  Quality: Good match")
                    elif quality['fair_match']:
                        print("  Quality: Fair match")
                    else:
                        print("  Quality: Poor match")
                    
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"Error: {e}")

def create_test_images():
    """Create simple test fingerprint images for demonstration"""
    print("\n=== Creating Test Images ===")
    
    import numpy as np
    from PIL import Image, ImageDraw
    
    # Create a simple fingerprint-like pattern
    def create_fingerprint_pattern(size=256):
        # Create base image
        img = Image.new('L', (size, size), 255)
        draw = ImageDraw.Draw(img)
        
        # Draw ridge patterns
        for i in range(0, size, 4):
            # Horizontal ridges
            draw.line([(0, i), (size, i)], fill=0, width=1)
            
        # Add some minutiae points
        minutiae_points = [
            (50, 50), (100, 80), (150, 120), (200, 160),
            (80, 200), (120, 40), (180, 180), (40, 180)
        ]
        
        for point in minutiae_points:
            draw.ellipse([point[0]-2, point[1]-2, point[0]+2, point[1]+2], fill=0)
        
        return img
    
    # Create a fingerprint pattern
    img = create_fingerprint_pattern()
    
    # Save image
    img.save("fingerprint.png")
    
    print("Created test image: fingerprint.png")

if __name__ == "__main__":
    print("Fingerprint Minutiae Matching Example")
    print("=" * 40)
    
    # Create test images if they don't exist
    if not os.path.exists("fingerprint.png"):
        create_test_images()
    
    # Run examples
    example_direct_usage()
    example_api_usage()
    
    print("\n=== Usage Instructions ===")
    print("1. Direct usage: Use FingerprintsMatching.fingerprints_matching(image1_path, image2_path)")
    print("2. API usage: POST to /minutiaematch with one fingerprint file")
    print("3. The API will compare against the entire AWS S3 fingerprint dataset")
    print("4. Match scores range from 0.0 (no match) to 1.0 (perfect match)")
    print("5. Scores >= 0.8: Excellent match")
    print("6. Scores >= 0.6: Good match")
    print("7. Scores >= 0.4: Fair match")
    print("8. Scores < 0.4: Poor match") 