#!/usr/bin/env python3
"""
Advanced Minutiae Fingerprint Matching Example

This script demonstrates the advanced minutiae fingerprint matching endpoint
with multiple analysis techniques for extremely accurate fingerprint matching.

Features demonstrated:
- Ridge pattern analysis using Gabor filters
- Orientation field analysis
- Texture analysis using Local Binary Patterns
- Minutiae point detection (ridge endings and bifurcations)
- Core and delta point detection
- Multiple similarity metrics with weighted combination
"""

import requests
import json
import time
import os
from typing import Dict, Any, Optional
import base64
from pathlib import Path

class AdvancedMinutiaeMatcher:
    """Advanced minutiae fingerprint matching client"""
    
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
    
    def get_info(self) -> Dict[str, Any]:
        """Get API information"""
        try:
            response = self.session.get(f"{self.base_url}/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Info request failed: {e}")
            return {"error": str(e)}
    
    def advanced_minutiae_match(
        self,
        image_path: str,
        threshold: float = 0.7,
        batch_size: int = 50,
        max_results: int = 10,
        enable_ridge_analysis: bool = True,
        enable_orientation_analysis: bool = True,
        enable_texture_analysis: bool = True,
        enable_minutiae_detection: bool = True
    ) -> Dict[str, Any]:
        """
        Perform advanced minutiae fingerprint matching
        
        Args:
            image_path: Path to the fingerprint image
            threshold: Similarity threshold (0.0 to 1.0)
            batch_size: Number of images to process in each batch
            max_results: Maximum number of results to return
            enable_ridge_analysis: Enable ridge pattern analysis
            enable_orientation_analysis: Enable orientation field analysis
            enable_texture_analysis: Enable texture analysis
            enable_minutiae_detection: Enable minutiae point detection
        
        Returns:
            Dictionary containing matching results and analysis details
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}
            
            # Prepare the request
            url = f"{self.base_url}/advanced-minutiae-match"
            
            with open(image_path, 'rb') as f:
                files = {'fingerprint': f}
                data = {
                    'threshold': threshold,
                    'batch_size': batch_size,
                    'max_results': max_results,
                    'enable_ridge_analysis': enable_ridge_analysis,
                    'enable_orientation_analysis': enable_orientation_analysis,
                    'enable_texture_analysis': enable_texture_analysis,
                    'enable_minutiae_detection': enable_minutiae_detection
                }
                
                print(f"🔍 Performing advanced minutiae matching...")
                print(f"   📁 Image: {image_path}")
                print(f"   🎯 Threshold: {threshold}")
                print(f"   📊 Analysis techniques:")
                print(f"      - Ridge analysis: {'✅' if enable_ridge_analysis else '❌'}")
                print(f"      - Orientation analysis: {'✅' if enable_orientation_analysis else '❌'}")
                print(f"      - Texture analysis: {'✅' if enable_texture_analysis else '❌'}")
                print(f"      - Minutiae detection: {'✅' if enable_minutiae_detection else '❌'}")
                
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
    
    def compare_analysis_techniques(
        self, 
        image_path: str,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Compare different analysis technique combinations
        
        Args:
            image_path: Path to the fingerprint image
            threshold: Similarity threshold
        
        Returns:
            Dictionary with comparison results
        """
        print(f"\n🔬 Comparing Analysis Techniques")
        print(f"   📁 Image: {image_path}")
        print(f"   🎯 Threshold: {threshold}")
        
        techniques = [
            {
                "name": "Full Analysis (All Techniques)",
                "config": {
                    "enable_ridge_analysis": True,
                    "enable_orientation_analysis": True,
                    "enable_texture_analysis": True,
                    "enable_minutiae_detection": True
                }
            },
            {
                "name": "Ridge + Orientation Analysis",
                "config": {
                    "enable_ridge_analysis": True,
                    "enable_orientation_analysis": True,
                    "enable_texture_analysis": False,
                    "enable_minutiae_detection": False
                }
            },
            {
                "name": "Minutiae + Texture Analysis",
                "config": {
                    "enable_ridge_analysis": False,
                    "enable_orientation_analysis": False,
                    "enable_texture_analysis": True,
                    "enable_minutiae_detection": True
                }
            },
            {
                "name": "Ridge Analysis Only",
                "config": {
                    "enable_ridge_analysis": True,
                    "enable_orientation_analysis": False,
                    "enable_texture_analysis": False,
                    "enable_minutiae_detection": False
                }
            },
            {
                "name": "Minutiae Detection Only",
                "config": {
                    "enable_ridge_analysis": False,
                    "enable_orientation_analysis": False,
                    "enable_texture_analysis": False,
                    "enable_minutiae_detection": True
                }
            }
        ]
        
        results = {}
        
        for technique in techniques:
            print(f"\n   🔍 Testing: {technique['name']}")
            
            result = self.advanced_minutiae_match(
                image_path=image_path,
                threshold=threshold,
                **technique['config']
            )
            
            if 'error' not in result:
                results[technique['name']] = {
                    'matches_found': len(result['results']['matches']),
                    'processing_time': result['results']['processing_time'],
                    'total_time': result['results']['total_time'],
                    'query_features': {
                        'keypoint_count': result['query']['keypoint_count'],
                        'minutiae_count': result['query']['minutiae_count'],
                        'ridge_features_count': result['query']['ridge_features_count']
                    }
                }
                
                if result['results']['matches']:
                    top_match = result['results']['matches'][0]
                    results[technique['name']]['top_match_score'] = top_match['similarity']
                    results[technique['name']]['top_match_quality'] = top_match['match_quality']
                else:
                    results[technique['name']]['top_match_score'] = 0.0
                    results[technique['name']]['top_match_quality'] = None
            else:
                results[technique['name']] = {'error': result['error']}
        
        return results
    
    def analyze_match_details(self, result: Dict[str, Any]) -> None:
        """Analyze and display detailed match information"""
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"\n📊 Advanced Minutiae Match Results")
        print(f"   ⏱️  Total Processing Time: {result['results']['total_time']:.2f}ms")
        print(f"   🔍 Images Processed: {result['results']['total_processed']}")
        print(f"   ✅ Matches Found: {len(result['results']['matches'])}")
        
        # Query analysis
        print(f"\n🔬 Query Analysis:")
        print(f"   📊 Keypoints: {result['query']['keypoint_count']}")
        print(f"   🎯 Minutiae Points: {result['query']['minutiae_count']}")
        print(f"   🏔️  Ridge Features: {result['query']['ridge_features_count']}")
        
        # Analysis techniques used
        techniques = result['results']['analysis_techniques_used']
        print(f"\n🔧 Analysis Techniques Used:")
        for technique, enabled in techniques.items():
            status = "✅" if enabled else "❌"
            print(f"   {status} {technique.replace('_', ' ').title()}")
        
        # Top matches
        if result['results']['matches']:
            print(f"\n🏆 Top Matches:")
            for i, match in enumerate(result['results']['matches'][:5]):
                quality_indicators = []
                for quality, is_true in match['match_quality'].items():
                    if is_true:
                        quality_indicators.append(quality.replace('_', ' ').title())
                
                print(f"   {i+1}. {match['filename']}")
                print(f"      Score: {match['similarity']:.3f} ({match['similarity']*100:.1f}%)")
                print(f"      Quality: {', '.join(quality_indicators)}")
                
                if 'analysis_details' in match:
                    details = match['analysis_details']
                    print(f"      Analysis: {details['keypoint_count']} keypoints, "
                          f"{details['minutiae_count']} minutiae, "
                          f"{details['ridge_features_count']} ridge features")
        else:
            print(f"\n❌ No matches found above threshold {result['configuration']['threshold']}")

def main():
    """Main function to demonstrate advanced minutiae matching"""
    
    # Initialize the matcher
    matcher = AdvancedMinutiaeMatcher()
    
    # Check API health
    print("🏥 Checking API health...")
    health = matcher.health_check()
    if health.get('status') != 'healthy':
        print(f"❌ API is not healthy: {health}")
        return
    
    print("✅ API is healthy!")
    
    # Get API info
    print("\n📋 Getting API information...")
    info = matcher.get_info()
    if 'error' not in info:
        print(f"   Service: {info['service']}")
        print(f"   Version: {info['version']}")
        print(f"   Description: {info['description']}")
    
    # Example image path (you can change this to your fingerprint image)
    image_path = "fingerprint.png"  # Change this to your image path
    
    if not os.path.exists(image_path):
        print(f"\n⚠️  Image file not found: {image_path}")
        print("   Please provide a valid fingerprint image path")
        return
    
    # Perform advanced minutiae matching with all techniques
    print(f"\n🚀 Performing Advanced Minutiae Matching")
    print("=" * 60)
    
    result = matcher.advanced_minutiae_match(
        image_path=image_path,
        threshold=0.7,
        batch_size=50,
        max_results=10,
        enable_ridge_analysis=True,
        enable_orientation_analysis=True,
        enable_texture_analysis=True,
        enable_minutiae_detection=True
    )
    
    # Analyze and display results
    matcher.analyze_match_details(result)
    
    # Compare different analysis techniques
    print(f"\n🔬 Technique Comparison")
    print("=" * 60)
    
    comparison = matcher.compare_analysis_techniques(image_path, threshold=0.7)
    
    print(f"\n📊 Comparison Results:")
    for technique_name, technique_result in comparison.items():
        print(f"\n   {technique_name}:")
        if 'error' in technique_result:
            print(f"      ❌ Error: {technique_result['error']}")
        else:
            print(f"      ✅ Matches: {technique_result['matches_found']}")
            print(f"      ⏱️  Processing Time: {technique_result['processing_time']:.2f}ms")
            print(f"      🎯 Top Match Score: {technique_result['top_match_score']:.3f}")
            
            if technique_result['top_match_quality']:
                quality = technique_result['top_match_quality']
                quality_indicators = [k.replace('_', ' ').title() for k, v in quality.items() if v]
                print(f"      🏆 Quality: {', '.join(quality_indicators)}")
    
    print(f"\n✨ Advanced Minutiae Matching Demo Complete!")

if __name__ == "__main__":
    main() 