#!/usr/bin/env python3
"""
Fingerprint Upload Example

This script demonstrates the fingerprint upload endpoint that uploads
fingerprint images to S3 with unique file keys and comprehensive metadata.

Features demonstrated:
- Unique file key generation
- Image preprocessing and validation
- Thumbnail generation
- Metadata extraction
- S3 upload with organized structure
"""

import requests
import json
import time
import os
from typing import Dict, Any, Optional
from pathlib import Path

class FingerprintUploader:
    """Fingerprint upload client"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def upload_fingerprint(
        self,
        image_path: str,
        user_id: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        enable_preprocessing: bool = True,
        generate_thumbnail: bool = True
    ) -> Dict[str, Any]:
        """
        Upload a fingerprint image to S3
        
        Args:
            image_path: Path to the fingerprint image
            user_id: Optional user ID for organization
            description: Optional description of the fingerprint
            category: Optional category for classification
            enable_preprocessing: Enable image preprocessing
            generate_thumbnail: Generate thumbnail version
        
        Returns:
            Dictionary containing upload results and metadata
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}
            
            # Prepare the request
            url = f"{self.base_url}/upload-fingerprint"
            
            with open(image_path, 'rb') as f:
                files = {'fingerprint': f}
                data = {
                    'user_id': user_id,
                    'description': description,
                    'category': category,
                    'enable_preprocessing': enable_preprocessing,
                    'generate_thumbnail': generate_thumbnail
                }
                
                # Remove None values
                data = {k: v for k, v in data.items() if v is not None}
                
                print(f"📤 Uploading fingerprint to S3...")
                print(f"   📁 Image: {image_path}")
                print(f"   👤 User ID: {user_id or 'Anonymous'}")
                print(f"   📝 Description: {description or 'No description'}")
                print(f"   🏷️  Category: {category or 'Uncategorized'}")
                print(f"   🔧 Preprocessing: {'✅' if enable_preprocessing else '❌'}")
                print(f"   🖼️  Thumbnail: {'✅' if generate_thumbnail else '❌'}")
                
                start_time = time.time()
                response = self.session.post(url, files=files, data=data)
                end_time = time.time()
                
                response.raise_for_status()
                result = response.json()
                
                # Add timing information
                result['client_processing_time'] = (end_time - start_time) * 1000
                
                return result
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}
    
    def batch_upload(
        self,
        image_directory: str,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        enable_preprocessing: bool = True,
        generate_thumbnail: bool = True
    ) -> Dict[str, Any]:
        """
        Upload multiple fingerprint images from a directory
        
        Args:
            image_directory: Directory containing fingerprint images
            user_id: Optional user ID for organization
            category: Optional category for classification
            enable_preprocessing: Enable image preprocessing
            generate_thumbnail: Generate thumbnail versions
        
        Returns:
            Dictionary containing batch upload results
        """
        if not os.path.exists(image_directory):
            return {"error": f"Directory not found: {image_directory}"}
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        image_files = []
        
        for file in os.listdir(image_directory):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(image_directory, file))
        
        if not image_files:
            return {"error": f"No image files found in directory: {image_directory}"}
        
        print(f"📁 Found {len(image_files)} images to upload")
        print(f"   📂 Directory: {image_directory}")
        print(f"   👤 User ID: {user_id or 'Anonymous'}")
        print(f"   🏷️  Category: {category or 'Uncategorized'}")
        
        results = {
            "successful_uploads": [],
            "failed_uploads": [],
            "total_files": len(image_files),
            "successful_count": 0,
            "failed_count": 0
        }
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n   📤 Uploading {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Generate description from filename
            filename = os.path.basename(image_path)
            description = f"Uploaded from {filename}"
            
            result = self.upload_fingerprint(
                image_path=image_path,
                user_id=user_id,
                description=description,
                category=category,
                enable_preprocessing=enable_preprocessing,
                generate_thumbnail=generate_thumbnail
            )
            
            if 'error' in result:
                results["failed_uploads"].append({
                    "file": image_path,
                    "error": result["error"]
                })
                results["failed_count"] += 1
                print(f"      ❌ Failed: {result['error']}")
            else:
                results["successful_uploads"].append({
                    "file": image_path,
                    "file_key": result["upload"]["file_key"],
                    "thumbnail_key": result["upload"]["thumbnail_key"],
                    "metadata_key": result["upload"]["metadata_key"]
                })
                results["successful_count"] += 1
                print(f"      ✅ Success: {result['upload']['file_key']}")
        
        return results
    
    def analyze_upload_result(self, result: Dict[str, Any]) -> None:
        """Analyze and display upload result details"""
        if 'error' in result:
            print(f"❌ Upload Error: {result['error']}")
            return
        
        print(f"\n📊 Upload Results")
        print(f"   ✅ Success: {result['success']}")
        print(f"   ⏱️  Processing Time: {result['upload']['processing_time']:.2f}ms")
        
        # Upload details
        upload = result['upload']
        print(f"\n📤 Upload Details:")
        print(f"   🔑 File Key: {upload['file_key']}")
        print(f"   🖼️  Thumbnail Key: {upload['thumbnail_key']}")
        print(f"   📋 Metadata Key: {upload['metadata_key']}")
        print(f"   📁 Original Filename: {upload['original_filename']}")
        print(f"   📏 File Size: {upload['file_size']} bytes ({upload['file_size']/1024:.1f} KB)")
        print(f"   🕒 Upload Timestamp: {upload['upload_timestamp']}")
        
        # Image information
        image_info = result['image_info']
        print(f"\n🖼️  Image Information:")
        print(f"   📐 Dimensions: {image_info['width']} x {image_info['height']} pixels")
        print(f"   📊 Aspect Ratio: {image_info['aspect_ratio']}")
        print(f"   💾 File Size: {image_info['file_size_mb']} MB")
        
        # Metadata
        metadata = result['metadata']
        print(f"\n📋 Metadata:")
        print(f"   👤 User ID: {metadata.get('user_id', 'Anonymous')}")
        print(f"   📝 Description: {metadata.get('description', 'No description')}")
        print(f"   🏷️  Category: {metadata.get('category', 'Uncategorized')}")
        print(f"   🔧 Preprocessing: {'✅' if metadata['preprocessing_applied'] else '❌'}")
        print(f"   🖼️  Thumbnail: {'✅' if metadata['thumbnail_generated'] else '❌'}")
        
        # Image statistics
        stats = metadata['image_statistics']
        print(f"\n📊 Image Statistics:")
        print(f"   📈 Mean Intensity: {stats['mean_intensity']}")
        print(f"   📉 Std Intensity: {stats['std_intensity']}")
        print(f"   🔽 Min Intensity: {stats['min_intensity']}")
        print(f"   🔼 Max Intensity: {stats['max_intensity']}")
        print(f"   🎯 Sharpness: {stats['sharpness']}")
        print(f"   📊 Entropy: {stats['entropy']}")
        
        # File hash
        print(f"\n🔐 File Hash: {metadata['file_hash'][:16]}...")

def main():
    """Main function to demonstrate fingerprint upload"""
    
    # Initialize the uploader
    uploader = FingerprintUploader()
    
    # Check API health
    print("🏥 Checking API health...")
    health = uploader.health_check()
    if health.get('status') != 'healthy':
        print(f"❌ API is not healthy: {health}")
        return
    
    print("✅ API is healthy!")
    
    # Example image path (you can change this to your fingerprint image)
    image_path = "fingerprint.png"  # Change this to your image path
    
    if not os.path.exists(image_path):
        print(f"\n⚠️  Image file not found: {image_path}")
        print("   Please provide a valid fingerprint image path")
        return
    
    # Upload single fingerprint
    print(f"\n🚀 Uploading Single Fingerprint")
    print("=" * 60)
    
    result = uploader.upload_fingerprint(
        image_path=image_path,
        user_id="user123",
        description="Right thumb fingerprint",
        category="biometric",
        enable_preprocessing=True,
        generate_thumbnail=True
    )
    
    # Analyze and display results
    uploader.analyze_upload_result(result)
    
    # Example batch upload (uncomment if you have a directory of images)
    """
    print(f"\n📁 Batch Upload Example")
    print("=" * 60)
    
    batch_result = uploader.batch_upload(
        image_directory="fingerprints/",
        user_id="user123",
        category="biometric",
        enable_preprocessing=True,
        generate_thumbnail=True
    )
    
    print(f"\n📊 Batch Upload Summary:")
    print(f"   📁 Total Files: {batch_result['total_files']}")
    print(f"   ✅ Successful: {batch_result['successful_count']}")
    print(f"   ❌ Failed: {batch_result['failed_count']}")
    """
    
    print(f"\n✨ Fingerprint Upload Demo Complete!")

if __name__ == "__main__":
    main() 