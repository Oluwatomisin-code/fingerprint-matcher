#!/usr/bin/env python3
"""
Comprehensive Minutiae Fingerprint Matching Example

This script demonstrates the comprehensive minutiae fingerprint matching endpoint
that integrates MinutiaeFeature, SIFT, and AWS S3 for robust fingerprint matching.

Features demonstrated:
- Minutiae extraction (terminations and bifurcations)
- SIFT feature matching
- Configurable distance and angle thresholds
- AWS S3 integration for large-scale matching
- Batch processing for performance
"""

import requests
import json
import time
import os
from typing import Dict, Any, Optional
import base64
from pathlib import Path

class ComprehensiveMinutiaeMatcher:
    """Comprehensive minutiae fingerprint matching client"""
    
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
    
    def comprehensive_minutiae_match(
        self,
        image_path: str,
        threshold: float = 0.4,
        batch_size: int = 50,
        max_results: int = 10,
        enable_minutiae_matching: bool = True,
        enable_sift_matching: bool = True,
        dist_threshold: int = 15,
        angle_threshold: int = 20,
        ratio_threshold: float = 0.75
    ) -> Dict[str, Any]:
        """
        Perform comprehensive minutiae fingerprint matching
        
        Args:
            image_path: Path to the fingerprint image
            threshold: Similarity threshold (0.0 to 1.0)
            batch_size: Number of images to process in each batch
            max_results: Maximum number of results to return
            enable_minutiae_matching: Enable minutiae-based matching
            enable_sift_matching: Enable SIFT-based matching
            dist_threshold: Distance threshold for minutiae matching
            angle_threshold: Angle threshold for minutiae matching
            ratio_threshold: Ratio threshold for SIFT matching
        
        Returns:
            Dictionary containing matching results and analysis details
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}
            
            # Prepare the request
            url = f"{self.base_url}/comprehensive-minutiae-match"
            
            with open(image_path, 'rb') as f:
                files = {'fingerprint': f}
                data = {
                    'threshold': threshold,
                    'batch_size': batch_size,
                    'max_results': max_results,
                    'enable_minutiae_matching': enable_minutiae_matching,
                    'enable_sift_matching': enable_sift_matching,
                    'dist_threshold': dist_threshold,
                    'angle_threshold': angle_threshold,
                    'ratio_threshold': ratio_threshold
                }
                
                print(f"🔍 Performing comprehensive minutiae matching...")
                print(f"   📁 Image: {image_path}")
                print(f"   🎯 Threshold: {threshold}")
                print(f"   📊 Matching techniques:")
                print(f"      - Minutiae matching: {'✅' if enable_minutiae_matching else '❌'}")
                print(f"      - SIFT matching: {'✅' if enable_sift_matching else '❌'}")
                print(f"   ⚙️  Configuration:")
                print(f"      - Distance threshold: {dist_threshold}")
                print(f"      - Angle threshold: {angle_threshold}")
                print(f"      - Ratio threshold: {ratio_threshold}")
                
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
    
    def compare_matching_techniques(
        self, 
        image_path: str,
        threshold: float = 0.4
    ) -> Dict[str, Any]:
        """
        Compare different matching technique combinations
        
        Args:
            image_path: Path to the fingerprint image
            threshold: Similarity threshold
        
        Returns:
            Dictionary with comparison results
        """
        print(f"\n🔬 Comparing Matching Techniques")
        print(f"   📁 Image: {image_path}")
        print(f"   🎯 Threshold: {threshold}")
        
        techniques = [
            {
                "name": "Comprehensive (Minutiae + SIFT)",
                "config": {
                    "enable_minutiae_matching": True,
                    "enable_sift_matching": True
                }
            },
            {
                "name": "Minutiae Only",
                "config": {
                    "enable_minutiae_matching": True,
                    "enable_sift_matching": False
                }
            },
            {
                "name": "SIFT Only",
                "config": {
                    "enable_minutiae_matching": False,
                    "enable_sift_matching": True
                }
            }
        ]
        
        results = {}
        
        for technique in techniques:
            print(f"\n   🔍 Testing: {technique['name']}")
            
            result = self.comprehensive_minutiae_match(
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
                        'minutiae_count': result['query']['minutiae_count'],
                        'termination_count': result['query']['termination_count'],
                        'bifurcation_count': result['query']['bifurcation_count']
                    }
                }
                
                if result['results']['matches']:
                    top_match = result['results']['matches'][0]
                    results[technique['name']]['top_match_score'] = top_match['similarity_score']
                    results[technique['name']]['minutiae_score'] = top_match.get('minutiae_score')
                    results[technique['name']]['sift_score'] = top_match.get('sift_score')
                    results[technique['name']]['match_quality'] = top_match['match_quality']
                else:
                    results[technique['name']]['top_match_score'] = 0.0
                    results[technique['name']]['minutiae_score'] = None
                    results[technique['name']]['sift_score'] = None
                    results[technique['name']]['match_quality'] = None
            else:
                results[technique['name']] = {'error': result['error']}
        
        return results
    
    def analyze_match_details(self, result: Dict[str, Any]) -> None:
        """Analyze and display detailed match information"""
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"\n📊 Comprehensive Minutiae Match Results")
        print(f"   ⏱️  Total Processing Time: {result['results']['total_time']:.2f}ms")
        print(f"   🔍 Images Processed: {result['results']['total_processed']}")
        print(f"   ✅ Matches Found: {len(result['results']['matches'])}")
        
        # Query analysis
        print(f"\n🔬 Query Analysis:")
        print(f"   🎯 Total Minutiae: {result['query']['minutiae_count']}")
        print(f"   📍 Terminations: {result['query']['termination_count']}")
        print(f"   🔀 Bifurcations: {result['query']['bifurcation_count']}")
        
        # Configuration
        config = result['configuration']
        print(f"\n⚙️  Configuration:")
        print(f"   🎯 Threshold: {config['threshold']}")
        print(f"   📦 Batch Size: {config['batch_size']}")
        print(f"   🔍 Minutiae Matching: {'✅' if config['enable_minutiae_matching'] else '❌'}")
        print(f"   🎯 SIFT Matching: {'✅' if config['enable_sift_matching'] else '❌'}")
        print(f"   📏 Distance Threshold: {config['dist_threshold']}")
        print(f"   📐 Angle Threshold: {config['angle_threshold']}")
        print(f"   📊 Ratio Threshold: {config['ratio_threshold']}")
        
        # Top matches
        if result['results']['matches']:
            print(f"\n🏆 Top Matches:")
            for i, match in enumerate(result['results']['matches'][:5]):
                quality_indicators = []
                for quality, is_true in match['match_quality'].items():
                    if is_true:
                        quality_indicators.append(quality.replace('_', ' ').title())
                
                print(f"   {i+1}. {match['filename']}")
                print(f"      Score: {match['similarity_score']:.3f} ({match['similarity_score']*100:.1f}%)")
                print(f"      Quality: {', '.join(quality_indicators)}")
                
                if match.get('minutiae_score') is not None:
                    print(f"      Minutiae Score: {match['minutiae_score']:.3f}")
                if match.get('sift_score') is not None:
                    print(f"      SIFT Score: {match['sift_score']:.3f}")
                
                print(f"      Target Minutiae: {match['target_minutiae_count']} "
                      f"({match['target_termination_count']} term, {match['target_bifurcation_count']} bif)")
        else:
            print(f"\n❌ No matches found above threshold {result['configuration']['threshold']}")

def main():
    """Main function to demonstrate comprehensive minutiae matching"""
    
    # Initialize the matcher
    matcher = ComprehensiveMinutiaeMatcher()
    
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
    
    # Perform comprehensive minutiae matching
    print(f"\n🚀 Performing Comprehensive Minutiae Matching")
    print("=" * 60)
    
    result = matcher.comprehensive_minutiae_match(
        image_path=image_path,
        threshold=0.4,
        batch_size=50,
        max_results=10,
        enable_minutiae_matching=True,
        enable_sift_matching=True,
        dist_threshold=15,
        angle_threshold=20,
        ratio_threshold=0.75
    )
    
    # Analyze and display results
    matcher.analyze_match_details(result)
    
    # Compare different matching techniques
    print(f"\n🔬 Technique Comparison")
    print("=" * 60)
    
    comparison = matcher.compare_matching_techniques(image_path, threshold=0.4)
    
    print(f"\n📊 Comparison Results:")
    for technique_name, technique_result in comparison.items():
        print(f"\n   {technique_name}:")
        if 'error' in technique_result:
            print(f"      ❌ Error: {technique_result['error']}")
        else:
            print(f"      ✅ Matches: {technique_result['matches_found']}")
            print(f"      ⏱️  Processing Time: {technique_result['processing_time']:.2f}ms")
            print(f"      🎯 Top Match Score: {technique_result['top_match_score']:.3f}")
            
            if technique_result.get('minutiae_score') is not None:
                print(f"      🔍 Minutiae Score: {technique_result['minutiae_score']:.3f}")
            if technique_result.get('sift_score') is not None:
                print(f"      🎯 SIFT Score: {technique_result['sift_score']:.3f}")
            
            if technique_result.get('match_quality'):
                quality = technique_result['match_quality']
                quality_indicators = [k.replace('_', ' ').title() for k, v in quality.items() if v]
                print(f"      🏆 Quality: {', '.join(quality_indicators)}")
    
    print(f"\n✨ Comprehensive Minutiae Matching Demo Complete!")

if __name__ == "__main__":
    main() 