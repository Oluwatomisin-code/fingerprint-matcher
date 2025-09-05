import os
import asyncio
import aiohttp
import base64
import time
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FingerprintClient:
    """Python client for the Fingerprint Matcher API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def match_from_file(self, file_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Match fingerprint from file"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Prepare form data
            data = aiohttp.FormData()
            
            # Add file
            with open(file_path, 'rb') as f:
                data.add_field('fingerprint', f, filename=Path(file_path).name)
            
            # Add options
            if options:
                if 'threshold' in options:
                    data.add_field('threshold', str(options['threshold']))
                if 'batch_size' in options:
                    data.add_field('batch_size', str(options['batch_size']))
                if 'max_results' in options:
                    data.add_field('max_results', str(options['max_results']))
            
            async with self.session.post(f"{self.base_url}/match", data=data) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {response.reason}")
                
                return await response.json()
                
        except Exception as error:
            logger.error(f"Error matching from file: {error}")
            raise
    
    async def match_from_base64(self, base64_image: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Match fingerprint from base64 image"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            payload = {
                'image': base64_image
            }
            
            if options:
                payload.update(options)
            
            async with self.session.post(
                f"{self.base_url}/match/base64",
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {response.reason}")
                
                return await response.json()
                
        except Exception as error:
            logger.error(f"Error matching from base64: {error}")
            raise
    
    async def match_from_buffer(self, buffer: bytes, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Match fingerprint from buffer"""
        try:
            base64_image = base64.b64encode(buffer).decode('utf-8')
            return await self.match_from_base64(base64_image, options)
        except Exception as error:
            logger.error(f"Error matching from buffer: {error}")
            raise
    
    async def get_health(self) -> Dict[str, Any]:
        """Get service health status"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {response.reason}")
                
                return await response.json()
                
        except Exception as error:
            logger.error(f"Error getting health status: {error}")
            raise
    
    async def get_info(self) -> Dict[str, Any]:
        """Get service information"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(f"{self.base_url}/info") as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {response.reason}")
                
                return await response.json()
                
        except Exception as error:
            logger.error(f"Error getting service info: {error}")
            raise
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(f"{self.base_url}/cache/stats") as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {response.reason}")
                
                return await response.json()
                
        except Exception as error:
            logger.error(f"Error getting cache stats: {error}")
            raise
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear cache"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(f"{self.base_url}/cache/clear") as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {response.reason}")
                
                return await response.json()
                
        except Exception as error:
            logger.error(f"Error clearing cache: {error}")
            raise

async def run_example():
    """Example usage of the fingerprint client"""
    logger.info("🔍 Fingerprint Matching Client Example")
    logger.info("=====================================")

    async with FingerprintClient() as client:
        try:
            # Check service health
            logger.info("\n1. Checking service health...")
            health = await client.get_health()
            logger.info(f"✅ Health status: {health}")

            # Get service info
            logger.info("\n2. Getting service information...")
            info = await client.get_info()
            logger.info(f"✅ Service info: {info}")

            # Get cache stats
            logger.info("\n3. Getting cache statistics...")
            cache_stats = await client.get_cache_stats()
            logger.info(f"✅ Cache stats: {cache_stats}")

            # Example: Match fingerprint from file
            sample_image_path = Path(__file__).parent.parent / "sample_fingerprint.jpg"

            if sample_image_path.exists():
                logger.info("\n4. Matching fingerprint from file...")
                result = await client.match_from_file(str(sample_image_path), {
                    'threshold': 0.7,
                    'max_results': 5
                })

                logger.info("✅ Match results:")
                logger.info(f"   - Total processed: {result['results']['total_processed']}")
                logger.info(f"   - Processing time: {result['results']['processing_time']:.0f}ms")
                logger.info(f"   - Matches found: {len(result['results']['matches'])}")

                if result['results']['matches']:
                    logger.info("\n🏆 Top matches:")
                    for i, match in enumerate(result['results']['matches'][:3]):
                        logger.info(f"   {i + 1}. {match['key']} - Similarity: {match['similarity'] * 100:.2f}%")
            else:
                logger.info("\n⚠️  Sample image not found. Skipping file match test.")

            # Example: Match fingerprint from base64
            logger.info("\n5. Testing base64 matching...")
            test_image_buffer = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            )
            base64_result = await client.match_from_base64(
                base64.b64encode(test_image_buffer).decode('utf-8'),
                {
                    'threshold': 0.8,
                    'max_results': 3
                }
            )

            logger.info("✅ Base64 match test completed")
            logger.info(f"   - Processing time: {base64_result['results']['processing_time']:.0f}ms")

        except Exception as error:
            logger.error(f"❌ Example failed: {error}")

            if "ECONNREFUSED" in str(error):
                logger.info("\n💡 Make sure the fingerprint matcher server is running:")
                logger.info("   python main.py")

async def run_performance_test():
    """Performance test"""
    logger.info("\n⚡ Performance Test")
    logger.info("==================")

    async with FingerprintClient() as client:
        try:
            sample_image_path = Path(__file__).parent.parent / "sample_fingerprint.jpg"

            if not sample_image_path.exists():
                logger.info("⚠️  Sample image not found. Skipping performance test.")
                return

            iterations = 3
            times = []

            for i in range(iterations):
                logger.info(f"\n🧪 Running iteration {i + 1}/{iterations}...")

                start_time = time.time()
                result = await client.match_from_file(str(sample_image_path), {
                    'threshold': 0.7,
                    'max_results': 10
                })
                end_time = time.time()

                times.append((end_time - start_time) * 1000)

                logger.info(f"   - Processing time: {result['results']['processing_time']:.0f}ms")
                logger.info(f"   - Total time: {(end_time - start_time) * 1000:.0f}ms")
                logger.info(f"   - Matches found: {len(result['results']['matches'])}")

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            logger.info("\n📊 Performance Summary:")
            logger.info(f"   - Average time: {avg_time:.2f}ms")
            logger.info(f"   - Min time: {min_time:.0f}ms")
            logger.info(f"   - Max time: {max_time:.0f}ms")
            logger.info(f"   - Standard deviation: {(sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5:.2f}ms")

        except Exception as error:
            logger.error(f"❌ Performance test failed: {error}")

async def main():
    """Main function to run examples"""
    await run_example()
    await run_performance_test()

if __name__ == "__main__":
    asyncio.run(main()) 