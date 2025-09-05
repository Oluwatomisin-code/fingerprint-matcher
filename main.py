import os
import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
import base64
from datetime import datetime
import cv2
import numpy as np
import uuid
import hashlib
from pathlib import Path
import aiohttp
from skimage.morphology import skeletonize
from skimage.filters import threshold_local
from scipy.spatial import cKDTree


from fingerprint_matcher import FingerprintMatcher
from fingerprints_matching import FingerprintsMatching
from MinutiaeFeature import MinutiaeExtractor, MinutiaeMatcher
from Extractor import SIFTMatcher
import tempfile
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fingerprint Matcher API",
    description="High-performance fingerprint matching using OpenCV and AWS S3",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize fingerprint matcher
fingerprint_matcher: Optional[FingerprintMatcher] = None

async def minutiae_find_matches(query_image_buffer: bytes, batch_size: Optional[int] = None) -> Dict[str, Any]:
    """Find matches for a query fingerprint using minutiae-based approach against AWS S3 dataset"""
    try:
        start_time = datetime.now()
        
        # Save query image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(query_image_buffer)
            temp_file.flush()
            query_path = temp_file.name
        
        try:
            # Extract query features using minutiae approach
            query_features = await minutiae_extract_features(query_image_buffer)
            query_keypoint_count = len(query_features['keypoints']) if query_features and query_features.get('keypoints') else 0
            
            # Check if we have valid query features
            if not query_features:
                logger.error("Failed to extract features from query image")
                return {
                    "matches": [],
                    "total_processed": 0,
                    "processing_time": 0,
                    "query_keypoint_count": 0
                }
            
            # Get all fingerprint keys from S3
            logger.info("Getting fingerprint keys from S3...")
            all_keys = await fingerprint_matcher.get_all_fingerprint_keys()
            logger.info(f"Received keys: {type(all_keys)}, length: {len(all_keys) if all_keys is not None else 'None'}")
            
            # Handle None or empty results
            if all_keys is None:
                logger.warning("get_all_fingerprint_keys returned None")
                all_keys = []
            
            if not all_keys or len(all_keys) == 0:
                logger.warning("No fingerprint keys found in S3 bucket")
                return {
                    "matches": [],
                    "total_processed": 0,
                    "processing_time": 0,
                    "query_keypoint_count": query_keypoint_count
                }
            
            # Ensure all_keys is a list
            if not isinstance(all_keys, list):
                logger.error(f"Expected list of keys, got {type(all_keys)}")
                return {
                    "matches": [],
                    "total_processed": 0,
                    "processing_time": 0,
                    "query_keypoint_count": query_keypoint_count
                }
            
            total_processed = 0
            matches = []
            
            # Set default batch size if not provided
            if batch_size is None:
                batch_size = 50
            
            # Process in batches
            # Final safety check to ensure all_keys is a valid list
            if not isinstance(all_keys, list):
                logger.error(f"all_keys is not a list: {type(all_keys)}")
                return {
                    "matches": [],
                    "total_processed": 0,
                    "processing_time": 0,
                    "query_keypoint_count": query_keypoint_count
                }
            
            for i in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[i:i + batch_size]
                
                # Process batch concurrently
                tasks = []
                for key in batch_keys:
                    task = minutiae_process_single_image(key, query_path, query_features)
                    tasks.append(task)
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect valid results
                for result in batch_results:
                    if isinstance(result, dict) and result.get('similarity', 0) >= fingerprint_matcher.match_threshold:
                        matches.append(result)
                    total_processed += 1
            
            # Sort matches by similarity score (descending)
            matches.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "matches": matches,
                "total_processed": total_processed,
                "processing_time": processing_time,
                "query_keypoint_count": query_keypoint_count
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(query_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary query file: {e}")
                
    except Exception as error:
        logger.error(f"Error in minutiae_find_matches: {error}")
        raise

# def _find_nbis_binaries() -> Dict[str, Optional[str]]:
#     """Locate NBIS binaries 'mindtct' and 'bozorth3' with fallbacks.

#     Order:
#     - Env vars NBIS_MINDTCT, NBIS_BOZORTH3
#     - PATH (shutil.which)
#     - Common local install at ~/nbis/bin
#     - Homebrew locations (/opt/homebrew/bin, /usr/local/bin)
#     """
#     # Env overrides
#     env_mindtct = os.getenv("NBIS_MINDTCT_PATH")
#     env_bozorth3 = os.getenv("NBIS_BOZORTH3_PATH")
#     if env_mindtct and os.path.isfile(env_mindtct):
#         mindtct_path = env_mindtct
#     else:
#         mindtct_path = shutil.which("mindtct")
#     if env_bozorth3 and os.path.isfile(env_bozorth3):
#         bozorth3_path = env_bozorth3
#     else:
#         bozorth3_path = shutil.which("bozorth3")

#     # Fallbacks
#     nbis_local = os.path.expanduser("~/nbis/bin")
#     for candidate in [mindtct_path, os.path.join(nbis_local, "mindtct"), "/opt/homebrew/bin/mindtct", "/usr/local/bin/mindtct"]:
#         if candidate and os.path.isfile(candidate):
#             mindtct_path = candidate
#             break
#     for candidate in [bozorth3_path, os.path.join(nbis_local, "bozorth3"), "/opt/homebrew/bin/bozorth3", "/usr/local/bin/bozorth3"]:
#         if candidate and os.path.isfile(candidate):
#             bozorth3_path = candidate
#             break

#     logger.info(f"NBIS paths -> mindtct: {mindtct_path}, bozorth3: {bozorth3_path}")
#     return {"mindtct": mindtct_path, "bozorth3": bozorth3_path}

# def _save_image_to_pgm(image_bytes: bytes, target_path: str) -> None:
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise Exception("Failed to decode image for NBIS processing")
#     ok = cv2.imwrite(target_path, img)
#     if not ok:
#         raise Exception("Failed to write temporary PGM image for NBIS")

# def _run_mindtct(mindtct_path: str, pgm_path: str, out_prefix: str) -> str:
#     """Run NBIS mindtct to generate .xyt template. Returns path to .xyt."""
#     if not mindtct_path or not os.path.isfile(mindtct_path):
#         raise Exception("NBIS mindtct binary not found. Set NBIS_MINDTCT_PATH or add to PATH.")
#     result = subprocess.run([mindtct_path, pgm_path, out_prefix], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if result.returncode != 0:
#         raise Exception(f"mindtct failed: {result.stderr.decode(errors='ignore')}")
#     xyt_path = f"{out_prefix}.xyt"
#     if not os.path.exists(xyt_path):
#         raise Exception("mindtct did not produce .xyt output")
#     return xyt_path

# def _run_bozorth3(bozorth3_path: str, xyt_query: str, xyt_candidate: str) -> int:
#     """Run NBIS bozorth3 to compute match score. Returns integer score."""
#     if not bozorth3_path or not os.path.isfile(bozorth3_path):
#         raise Exception("NBIS bozorth3 binary not found. Set NBIS_BOZORTH3_PATH or add to PATH.")
#     result = subprocess.run([bozorth3_path, xyt_query, xyt_candidate], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if result.returncode != 0:
#         # bozorth3 sometimes returns 0 with no output; still parse stdout
#         stderr_text = result.stderr.decode(errors='ignore').strip()
#         if stderr_text:
#             logger.warning(f"bozorth3 warning: {stderr_text}")
#     out = result.stdout.decode(errors='ignore').strip()
#     try:
#         return int(out) if out else 0
#     except ValueError:
#         return 0

# def _write_pgm_p5(image_gray: np.ndarray, path: str) -> None:
#     """Write an 8-bit grayscale image to binary PGM (P5) that NBIS reliably reads."""
#     if image_gray.dtype != np.uint8:
#         image_gray = image_gray.astype(np.uint8)
#     rows, cols = image_gray.shape[:2]
#     header = f"P5\n{cols} {rows}\n255\n".encode("ascii")
#     with open(path, 'wb') as f:
#         f.write(header)
#         f.write(image_gray.tobytes())

# def _write_pgm_p2(image_gray: np.ndarray, path: str) -> None:
#     """Write an 8-bit grayscale image to ASCII PGM (P2) as another NBIS-compatible option."""
#     if image_gray.dtype != np.uint8:
#         image_gray = image_gray.astype(np.uint8)
#     rows, cols = image_gray.shape[:2]
#     with open(path, 'w', newline='\n') as f:
#         f.write("P2\n")
#         f.write(f"{cols} {rows}\n")
#         f.write("255\n")
#         # Write pixels: 16 values per line to keep file reasonable
#         count = 0
#         for y in range(rows):
#             for x in range(cols):
#                 f.write(str(int(image_gray[y, x])))
#                 count += 1
#                 if count % 16 == 0:
#                     f.write("\n")
#                 else:
#                     f.write(" ")
#         if count % 16 != 0:
#             f.write("\n")


# async def _nbis_prepare_query_template(query_bytes: bytes, mindtct_path: str) -> str:
#     with tempfile.TemporaryDirectory() as tmpdir:
#         pass
#     # Decode bytes to grayscale
#     nparr = np.frombuffer(query_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise Exception("Failed to decode image for NBIS template")
#     img = cv2.resize(img, (512, 512))
#     # First: write ASCII PGM (P2), then binary PGM (P5)
#     for writer, label in [( _write_pgm_p2, 'PGM(P2)'), ( _write_pgm_p5, 'PGM(P5)')]:
#         fd, pgm_path = tempfile.mkstemp(suffix='.pgm')
#         os.close(fd)
#         try:
#             writer(img, pgm_path)
#             out_prefix = pgm_path[:-4]
#             xyt_path = await asyncio.to_thread(_run_mindtct, mindtct_path, pgm_path, out_prefix)
#             return xyt_path
#         except Exception as e:
#             logger.warning(f"NBIS: mindtct failed on {label}: {e}")

#     # Then try PNG and BMP fallbacks
#     for ext in ['.png', '.bmp']:
#         fd2, img_path = tempfile.mkstemp(suffix=ext)
#         os.close(fd2)
#         if not cv2.imwrite(img_path, img):
#             continue
#         out_prefix2 = img_path[:-len(ext)]
#         try:
#             xyt_path = await asyncio.to_thread(_run_mindtct, mindtct_path, img_path, out_prefix2)
#             return xyt_path
#         except Exception as e:
#             logger.warning(f"NBIS: mindtct failed on {ext}: {e}")
#             continue
#     raise Exception("NBIS: failed to generate template from image using PGM/PNG/BMP (no raw mode available)")

# async def _nbis_prepare_query_templates_with_rotations(query_bytes: bytes, mindtct_path: str, angles: List[int]) -> List[str]:
#     templates: List[str] = []
#     # Decode once
#     nparr = np.frombuffer(query_bytes, np.uint8)
#     base = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
#     if base is None:
#         return templates
#     base = cv2.resize(base, (512, 512))
#     h, w = base.shape
#     center = (w // 2, h // 2)
#     for angle in angles:
#         if angle == 0:
#             img = base
#         else:
#             M = cv2.getRotationMatrix2D(center, angle, 1.0)
#             img = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
#         success = False
#         # PGM writers first: P2 then P5
#         for writer, label in [( _write_pgm_p2, 'PGM(P2)'), ( _write_pgm_p5, 'PGM(P5)')]:
#             fd_pgm, pgm_path = tempfile.mkstemp(suffix='.pgm')
#             os.close(fd_pgm)
#             try:
#                 writer(img, pgm_path)
#                 out_prefix = pgm_path[:-4]
#                 xyt_path = await asyncio.to_thread(_run_mindtct, mindtct_path, pgm_path, out_prefix)
#                 templates.append(xyt_path)
#                 success = True
#                 break
#             except Exception as e:
#                 logger.warning(f"NBIS: mindtct failed for rotation {angle} on {label}: {e}")

#         # PNG/BMP fallbacks
#         if not success:
#             for ext in ['.png', '.bmp']:
#                 fd, path = tempfile.mkstemp(suffix=ext)
#                 os.close(fd)
#                 if not cv2.imwrite(path, img):
#                     continue
#                 out_prefix2 = path[:-len(ext)]
#                 try:
#                     xyt_path = await asyncio.to_thread(_run_mindtct, mindtct_path, path, out_prefix2)
#                     templates.append(xyt_path)
#                     success = True
#                     break
#                 except Exception as e:
#                     logger.warning(f"NBIS: mindtct failed for rotation {angle} on {ext}: {e}")
#                     continue
#         if not success:
#             logger.warning(f"NBIS: all format fallbacks failed for rotation {angle}")
#     return templates

# async def _nbis_get_or_make_s3_template(key: str, mindtct_path: str) -> Optional[str]:
#     """Try to fetch a .xyt from S3 alongside the image; otherwise, download image and generate template."""
#     # Try candidate template keys
#     candidates = [
#         f"{key}.xyt",
#         str(Path(key).with_suffix('.xyt'))
#     ]
#     for tkey in candidates:
#         try:
#             obj = fingerprint_matcher.s3_client.get_object(Bucket=fingerprint_matcher.bucket_name, Key=tkey)
#             data = obj['Body'].read()
#             fd, path = tempfile.mkstemp(suffix='.xyt')
#             os.close(fd)
#             with open(path, 'wb') as f:
#                 f.write(data)
#             return path
#         except Exception:
#             continue

#     # Fallback: download the image and generate template
#     try:
#         response = fingerprint_matcher.s3_client.get_object(Bucket=fingerprint_matcher.bucket_name, Key=key)
#         image_buffer = response['Body'].read()
#     except Exception as e:
#         logger.error(f"NBIS: failed to download image {key}: {e}")
#         return None

#     fd_img, img_pgm = tempfile.mkstemp(suffix='.pgm')
#     os.close(fd_img)
#     try:
#         _save_image_to_pgm(image_buffer, img_pgm)
#         out_prefix = img_pgm[:-4]
#         xyt_path = await asyncio.to_thread(_run_mindtct, mindtct_path, img_pgm, out_prefix)
#         # Try to upload generated template for caching
#         try:
#             with open(xyt_path, 'rb') as f:
#                 fingerprint_matcher.s3_client.put_object(
#                     Bucket=fingerprint_matcher.bucket_name,
#                     Key=f"{key}.xyt",
#                     Body=f.read(),
#                     ContentType='application/octet-stream'
#                 )
#         except Exception as e:
#             logger.warning(f"NBIS: failed to upload xyt cache for {key}: {e}")
#         return xyt_path
#     except Exception as e:
#         logger.error(f"NBIS: mindtct failed for {key}: {e}")
#         return None

# async def _nbis_process_single_image(key: str, xyt_query: str, bozorth3_path: str, threshold_score: int) -> Optional[Dict[str, Any]]:
#     try:
#         mindtct_path = shutil.which("mindtct")
#         xyt_candidate = await _nbis_get_or_make_s3_template(key, mindtct_path)
#         if not xyt_candidate:
#             return None
#         score = await asyncio.to_thread(_run_bozorth3, bozorth3_path, xyt_query, xyt_candidate)
#         logger.info(f"nbis: key={key} score={score}")
#         if score >= threshold_score:
#             return {
#                 "key": key,
#                 "filename": key,
#                 "nbis_score": int(score),
#                 "similarity": float(min(score / 250.0, 1.0)),
#                 "match_quality": {
#                     "excellent_match": score >= 200,
#                     "very_good_match": score >= 150,
#                     "good_match": score >= 100,
#                     "fair_match": score >= 60,
#                     "poor_match": score < 60
#                 }
#             }
#         return None
#     except Exception as e:
#         logger.error(f"NBIS: error processing {key}: {e}")
#         return None

# async def _nbis_process_single_image_multi(
#     key: str,
#     xyt_queries: List[str],
#     bozorth3_path: str,
#     mindtct_path: str,
#     threshold_score: int
# ) -> Optional[Dict[str, Any]]:
#     try:
#         xyt_candidate = await _nbis_get_or_make_s3_template(key, mindtct_path)
#         if not xyt_candidate:
#             return None
#         best_score = 0
#         for xq in xyt_queries:
#             s = await asyncio.to_thread(_run_bozorth3, bozorth3_path, xq, xyt_candidate)
#             if s > best_score:
#                 best_score = s
#         logger.info(f"nbis-verify-dataset: key={key} best_score={best_score}")
#         if best_score >= threshold_score:
#             return {
#                 "key": key,
#                 "filename": key,
#                 "nbis_score": int(best_score),
#                 "similarity": float(min(best_score / 250.0, 1.0))
#             }
#         return None
#     except Exception as e:
#         logger.error(f"NBIS verify dataset: error processing {key}: {e}")
#         return None

# @app.post("/nbis-match")
# async def nbis_match(
#     fingerprint: UploadFile = File(...),
#     threshold_score: Optional[int] = Form(80),
#     batch_size: Optional[int] = Form(40),
#     max_results: Optional[int] = Form(10)
# ):
#     """Match using NBIS (MINDTCT + BOZORTH3) against S3 with per-comparison logging."""
#     try:
#         if not fingerprint_matcher:
#             raise HTTPException(status_code=500, detail="Fingerprint matcher not initialized")

#         # Check NBIS availability
#         bins = _find_nbis_binaries()
#         if not bins["mindtct"] or not bins["bozorth3"]:
#             raise HTTPException(status_code=500, detail="NBIS binaries not found. Please install 'mindtct' and 'bozorth3' and ensure they are on PATH.")

#         # Validate file
#         valid_types = ["image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif"]
#         if fingerprint.content_type not in valid_types:
#             raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, JPG, PNG, and TIFF are allowed.")

#         start_time = datetime.now()
#         file_content = await fingerprint.read()

#         # Prepare query template
#         xyt_query = await _nbis_prepare_query_template(file_content, bins["mindtct"]) 

#         # Keys from S3
#         all_keys = await fingerprint_matcher.get_all_fingerprint_keys()
#         if not all_keys:
#             return {
#                 "success": True,
#                 "query": {
#                     "file_size": fingerprint.size,
#                     "file_name": fingerprint.filename
#                 },
#                 "results": {
#                     "matches": [],
#                     "total_processed": 0,
#                     "processing_time": 0,
#                     "total_time": 0
#                 },
#                 "configuration": {
#                     "threshold_score": threshold_score,
#                     "batch_size": batch_size
#                 }
#             }

#         # Defaults
#         if batch_size is None:
#             batch_size = 40
#         if threshold_score is None:
#             threshold_score = 80

#         total_processed = 0
#         matches: List[Dict[str, Any]] = []

#         # Batch processing
#         for i in range(0, len(all_keys), batch_size):
#             batch_keys = all_keys[i:i + batch_size]
#             tasks = [
#                 _nbis_process_single_image(key, xyt_query, bins["bozorth3"], threshold_score)
#                 for key in batch_keys
#             ]
#             batch_results = await asyncio.gather(*tasks, return_exceptions=True)
#             for result in batch_results:
#                 if isinstance(result, dict) and result.get('nbis_score', 0) >= threshold_score:
#                     matches.append(result)
#                 total_processed += 1

#         # Sort by NBIS score desc
#         matches.sort(key=lambda x: x.get('nbis_score', 0), reverse=True)
#         if max_results and len(matches) > max_results:
#             matches = matches[:max_results]

#         processing_time = (datetime.now() - start_time).total_seconds() * 1000

#         response = {
#             "success": True,
#             "query": {
#                 "file_size": fingerprint.size,
#                 "file_name": fingerprint.filename
#             },
#             "results": {
#                 "matches": matches,
#                 "total_processed": total_processed,
#                 "processing_time": processing_time,
#                 "total_time": processing_time
#             },
#             "configuration": {
#                 "threshold_score": threshold_score,
#                 "batch_size": batch_size
#             }
#         }
#         logger.info(f"NBIS match response: {response}")
#         return response

#     except HTTPException:
#         raise
#     except Exception as error:
#         logger.error(f"Error in nbis-match endpoint: {error}")
#         raise HTTPException(status_code=500, detail=str(error))

# @app.post("/nbis-verify")
# async def nbis_verify(
#     fingerprint: UploadFile = File(...),
#     threshold_score: Optional[int] = Form(80),
#     rotation_range: Optional[int] = Form(0),
#     rotation_step: Optional[int] = Form(5),
#     batch_size: Optional[int] = Form(50),
#     max_results: Optional[int] = Form(10)
# ):
#     """Verify uploaded fingerprint against the entire S3 dataset using NBIS with optional rotation sweep.

#     Returns top matches across the bucket, similar to other endpoints (see line 881 behavior).
#     """
#     try:
#         valid = ["image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif"]
#         if fingerprint.content_type not in valid:
#             raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, JPG, PNG, and TIFF are allowed.")

#         bins = _find_nbis_binaries()
#         if not bins["mindtct"] or not bins["bozorth3"]:
#             raise HTTPException(status_code=500, detail="NBIS binaries not found. Install mindtct and bozorth3.")

#         start_time = datetime.now()
#         buf1 = await fingerprint.read()

#         rng = rotation_range or 0
#         step = rotation_step or 5
#         angles: List[int] = sorted(set([0] + ([a for a in range(-rng, rng + 1, step)] if rng > 0 else [])))

#         logger.info(f"nbis-verify: building probe templates with angles={angles}")
#         xyt_list_1 = await _nbis_prepare_query_templates_with_rotations(buf1, bins["mindtct"], angles)
#         if not xyt_list_1:
#             raise HTTPException(status_code=400, detail="Failed to generate templates for first image")
#         # Fetch all keys and evaluate in batches
#         logger.info("nbis-verify: fetching S3 keys")
#         all_keys = await fingerprint_matcher.get_all_fingerprint_keys()
#         if not all_keys:
#             return {
#                 "success": True,
#                 "query": {
#                     "file_size": fingerprint.size,
#                     "file_name": fingerprint.filename
#                 },
#                 "results": {
#                     "matches": [],
#                     "total_processed": 0,
#                     "processing_time": 0,
#                     "total_time": 0
#                 },
#                 "configuration": {
#                     "threshold_score": threshold_score,
#                     "rotation_range": rng,
#                     "rotation_step": step,
#                     "batch_size": batch_size or 50
#                 }
#             }

#         if batch_size is None:
#             batch_size = 50

#         total_processed = 0
#         matches: List[Dict[str, Any]] = []

#         for i in range(0, len(all_keys), batch_size):
#             batch_keys = all_keys[i:i + batch_size]
#             logger.info(f"nbis-verify: processing batch {i//batch_size+1} size={len(batch_keys)}")
#             tasks = [
#                 _nbis_process_single_image_multi(key, xyt_list_1, bins["bozorth3"], bins["mindtct"], threshold_score or 80)
#                 for key in batch_keys
#             ]
#             batch_results = await asyncio.gather(*tasks, return_exceptions=True)
#             for res in batch_results:
#                 if isinstance(res, dict):
#                     matches.append(res)
#                 total_processed += 1

#         # Sort by score and limit
#         matches.sort(key=lambda x: x.get('nbis_score', 0), reverse=True)
#         if max_results and len(matches) > max_results:
#             matches = matches[:max_results]

#         total_time = (datetime.now() - start_time).total_seconds() * 1000
#         response = {
#             "success": True,
#             "query": {
#                 "file_size": fingerprint.size,
#                 "file_name": fingerprint.filename
#             },
#             "results": {
#                 "matches": matches,
#                 "total_processed": total_processed,
#                 "processing_time": total_time,
#                 "total_time": total_time
#             },
#             "configuration": {
#                 "threshold_score": threshold_score,
#                 "rotation_range": rng,
#                 "rotation_step": step,
#                 "batch_size": batch_size
#             }
#         }
#         logger.info(f"NBIS verify dataset response: {response}")
#         return response

#     except HTTPException:
#         raise
#     except Exception as error:
#         logger.error(f"Error in nbis-verify endpoint: {error}")
#         raise HTTPException(status_code=500, detail=str(error))

async def minutiae_extract_features(image_buffer: bytes) -> Optional[Dict[str, Any]]:
    """Extract minutiae features from image buffer"""
    try:
        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(image_buffer)
            temp_file.flush()
            temp_path = temp_file.name
        
        try:
            # Load image and extract features
            img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error("Failed to load image from buffer")
                return None
            
            # Preprocess image
            img = FingerprintsMatching._preprocess_image(img)
            
            # Extract minutiae features
            minutiae = FingerprintsMatching._extract_minutiae(img)
            
            # Validate that we have valid features
            if not minutiae or not minutiae.get('keypoints'):
                logger.error("Failed to extract valid minutiae features")
                return None
            
            return minutiae
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary feature extraction file: {e}")
                
    except Exception as error:
        logger.error(f"Error in minutiae_extract_features: {error}")
        return None

async def minutiae_process_single_image(key: str, query_path: str, query_features: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Process a single image from S3 using minutiae-based matching"""
    try:
        # Download image from S3
        image_data = await fingerprint_matcher.download_and_process_image(key)
        if not image_data:
            return None
        
        # Since download_and_process_image returns features, not raw image buffer,
        # we need to get the actual image from S3 directly
        try:
            response = fingerprint_matcher.s3_client.get_object(Bucket=fingerprint_matcher.bucket_name, Key=key)
            image_buffer = response['Body'].read()
        except Exception as e:
            logger.error(f"Failed to download image {key}: {e}")
            return None
        
        # Save S3 image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(image_buffer)
            temp_file.flush()
            s3_image_path = temp_file.name
        
        try:
            # Perform minutiae matching
            try:
                match_score = FingerprintsMatching.fingerprints_matching(query_path, s3_image_path)
                logger.info(f"Match score: {match_score}, query_path: {query_path}, s3_image_path: {s3_image_path}")
            except Exception as e:
                logger.warning(f"Minutiae matching failed for {key}: {e}")
                return None
            
            if match_score >= fingerprint_matcher.match_threshold:
                return {
                    "key": key,
                    "similarity": match_score,
                    "filename": key,
                    "size": len(image_buffer),
                    "match_quality": {
                        "excellent_match": match_score >= 0.8,
                        "good_match": match_score >= 0.6,
                        "fair_match": match_score >= 0.4,
                        "poor_match": match_score < 0.4
                    }
                }
            
            return None
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(s3_image_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary S3 image file: {e}")
                
    except Exception as error:
        logger.error(f"Error processing image {key}: {error}")
        return None

@app.on_event("startup")
async def startup_event():
    """Initialize the fingerprint matcher on startup"""
    global fingerprint_matcher
    try:
        fingerprint_matcher = FingerprintMatcher({
            'aws_access_key_id': os.getenv("AWS_ACCESS_KEY_ID"),
            'aws_secret_access_key': os.getenv("AWS_SECRET_ACCESS_KEY"),
            'aws_region': os.getenv("AWS_REGION", "us-east-1"),
            'bucket_name': os.getenv("S3_BUCKET_NAME"),
            'match_threshold': float(os.getenv("MATCH_THRESHOLD", "0.7")),
            'max_concurrent_downloads': int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "10")),
            'feature_cache_size': int(os.getenv("FEATURE_CACHE_SIZE", "1000"))
        })
        logger.info("Fingerprint matcher initialized successfully")
    except Exception as error:
        logger.error(f"Failed to initialize fingerprint matcher: {error}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global fingerprint_matcher
    if fingerprint_matcher:
        fingerprint_matcher.cleanup()
        logger.info("Fingerprint matcher cleanup completed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "fingerprint-matcher"
    }

@app.get("/info")
async def get_info():
    """Get service information"""
    return {
        "service": "Fingerprint Matcher",
        "version": "1.0.0",
        "description": "High-performance fingerprint matching using OpenCV ORB features",
        "features": [
            "ORB feature extraction",
            "AWS S3 integration",
            "Batch processing",
            "Feature caching",
            "Configurable matching threshold"
        ],
        "configuration": {
            "match_threshold": fingerprint_matcher.match_threshold if fingerprint_matcher else None,
            "max_concurrent_downloads": fingerprint_matcher.max_concurrent_downloads if fingerprint_matcher else None,
            "feature_cache_size": fingerprint_matcher.feature_cache_size if fingerprint_matcher else None,
            "bucket_name": fingerprint_matcher.bucket_name if fingerprint_matcher else None
        }
    }

@app.post("/match")
async def match_fingerprint(
    fingerprint: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    batch_size: Optional[int] = Form(None),
    max_results: Optional[int] = Form(10)
):
    """Match fingerprint from uploaded file"""
    try:
        if not fingerprint_matcher:
            raise HTTPException(status_code=500, detail="Fingerprint matcher not initialized")

        # Validate file type
        if fingerprint.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG, JPG, PNG, and TIFF are allowed."
            )

        start_time = datetime.now()
        logger.info(f"Processing fingerprint match request - File size: {fingerprint.size} bytes")

        # Read file content
        file_content = await fingerprint.read()

        # Configure matcher with custom threshold if provided
        if threshold is not None:
            fingerprint_matcher.match_threshold = threshold

        # Find matches
        result = await fingerprint_matcher.find_matches(file_content, batch_size=batch_size)

        # Limit results if specified
        if max_results and len(result["matches"]) > max_results:
            result["matches"] = result["matches"][:max_results]

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "success": True,
            "query": {
                "file_size": fingerprint.size,
                "file_name": fingerprint.filename,
                "keypoint_count": result["query_keypoint_count"]
            },
            "results": {
                "matches": result["matches"],
                "total_processed": result["total_processed"],
                "processing_time": result["processing_time"],
                "total_time": total_time
            },
            "configuration": {
                "threshold": fingerprint_matcher.match_threshold,
                "batch_size": batch_size or 50
            }
        }

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error in match endpoint: {error}")
        raise HTTPException(status_code=500, detail=str(error))

@app.post("/match/base64")
async def match_fingerprint_base64(
    image: str,
    threshold: Optional[float] = None,
    batch_size: Optional[int] = None,
    max_results: int = 10
):
    """Match fingerprint from base64 encoded image"""
    try:
        if not fingerprint_matcher:
            raise HTTPException(status_code=500, detail="Fingerprint matcher not initialized")

        if not image:
            raise HTTPException(
                status_code=400,
                detail="No image data provided"
            )

        start_time = datetime.now()
        logger.info("Processing base64 fingerprint match request")

        # Decode base64 image
        try:
            image_buffer = base64.b64decode(image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        # Configure matcher with custom threshold if provided
        if threshold is not None:
            fingerprint_matcher.match_threshold = threshold

        # Find matches
        result = await fingerprint_matcher.find_matches(image_buffer, batch_size=batch_size)

        # Limit results if specified
        if max_results and len(result["matches"]) > max_results:
            result["matches"] = result["matches"][:max_results]

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "success": True,
            "query": {
                "image_size": len(image_buffer),
                "keypoint_count": result["query_keypoint_count"]
            },
            "results": {
                "matches": result["matches"],
                "total_processed": result["total_processed"],
                "processing_time": result["processing_time"],
                "total_time": total_time
            },
            "configuration": {
                "threshold": fingerprint_matcher.match_threshold,
                "batch_size": batch_size or 50
            }
        }

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error in base64 match endpoint: {error}")
        raise HTTPException(status_code=500, detail=str(error))

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    if not fingerprint_matcher:
        raise HTTPException(status_code=500, detail="Fingerprint matcher not initialized")

    cache_size = len(fingerprint_matcher.feature_cache)
    max_cache_size = fingerprint_matcher.feature_cache_size
    cache_utilization = (cache_size / max_cache_size * 100) if max_cache_size > 0 else 0

    return {
        "cache_size": cache_size,
        "max_cache_size": max_cache_size,
        "cache_utilization": f"{cache_utilization:.2f}%"
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear the feature cache"""
    if not fingerprint_matcher:
        raise HTTPException(status_code=500, detail="Fingerprint matcher not initialized")

    fingerprint_matcher.cleanup()
    return {
        "message": "Cache cleared successfully",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/minutiaematch")
async def minutiae_match(
    fingerprint: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    batch_size: Optional[int] = Form(None),
    max_results: Optional[int] = Form(10)
):
    """Match fingerprint from uploaded file against AWS S3 dataset using minutiae-based approach"""
    try:
        if not fingerprint_matcher:
            raise HTTPException(status_code=500, detail="Fingerprint matcher not initialized")

        # Validate file type
        if fingerprint.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG, JPG, PNG, and TIFF are allowed."
            )

        start_time = datetime.now()
        logger.info(f"Processing minutiae match request - File size: {fingerprint.size} bytes")

        # Read file content
        file_content = await fingerprint.read()

        # Configure matcher with custom threshold if provided
        if threshold is not None:
            fingerprint_matcher.match_threshold = threshold

        # Find matches using minutiae-based approach
        result = await minutiae_find_matches(file_content, batch_size=batch_size)
        logger.info(f"Minutiae match results: {result}")

        # Limit results if specified
        if max_results and len(result["matches"]) > max_results:
            result["matches"] = result["matches"][:max_results]

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "success": True,
            "query": {
                "file_size": fingerprint.size,
                "file_name": fingerprint.filename,
                "keypoint_count": result["query_keypoint_count"]
            },
            "results": {
                "matches": result["matches"],
                "total_processed": result["total_processed"],
                "processing_time": result["processing_time"],
                "total_time": total_time
            },
            "configuration": {
                "threshold": fingerprint_matcher.match_threshold,
                "batch_size": batch_size or 50
            }
        }

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error in minutiae match endpoint: {error}")
        raise HTTPException(status_code=500, detail=str(error))

# @app.post("/advanced-minutiae-match")
async def advanced_minutiae_match(
    fingerprint: UploadFile = File(...),
    threshold: Optional[float] = Form(0.65),
    batch_size: Optional[int] = Form(50),
    max_results: Optional[int] = Form(10),
    enable_ridge_analysis: bool = Form(True),
    enable_orientation_analysis: bool = Form(True),
    enable_texture_analysis: bool = Form(True),
    enable_minutiae_detection: bool = Form(True)
):
    """Advanced minutiae-based fingerprint matching with multiple analysis techniques"""
    try:
        if not fingerprint_matcher:
            raise HTTPException(status_code=500, detail="Fingerprint matcher not initialized")

        # Validate file type
        if fingerprint.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG, JPG, PNG, and TIFF are allowed."
            )

        start_time = datetime.now()
        logger.info(f"Processing advanced minutiae match request - File size: {fingerprint.size} bytes")

        # Read file content
        file_content = await fingerprint.read()

        # Configure matcher with custom threshold if provided
        if threshold is not None:
            fingerprint_matcher.match_threshold = threshold

        # Find matches using advanced minutiae-based approach
        result = await advanced_minutiae_find_matches(
            file_content, 
            batch_size=batch_size,
            enable_ridge_analysis=enable_ridge_analysis,
            enable_orientation_analysis=enable_orientation_analysis,
            enable_texture_analysis=enable_texture_analysis,
            enable_minutiae_detection=enable_minutiae_detection
        )

        # Limit results if specified
        if max_results and len(result["matches"]) > max_results:
            result["matches"] = result["matches"][:max_results]

        # Additional step: If we have matches, get the first match and validate polling unit officer
        officer_validation_result = None
        if result["matches"] and len(result["matches"]) > 0:
            first_match = result["matches"][0]
            match_filename = first_match.get("filename") or first_match.get("key", "")
            
            if match_filename:
                # Split by underscore and get the first part as UUID
                uuid_part = match_filename.split("_")[0]
                
                try:
                    # Make request to validate polling unit officer
                    async with aiohttp.ClientSession() as session:
                        validation_url = f"http://localhost:7500/validate_polling_unit_officer_by_uuid"
                        payload = {"uuid": uuid_part}
                        
                        async with session.post(validation_url, json=payload) as response:
                            if response.status == 200:
                                officer_validation_result = await response.json()
                            else:
                                logger.warning(f"Failed to validate officer: {response.status}")
                                officer_validation_result = {"error": f"Validation failed with status {response.status}"}
                except Exception as validation_error:
                    logger.error(f"Error validating polling unit officer: {validation_error}")
                    officer_validation_result = {"error": str(validation_error)}

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        response = {
            "success": True,
            "query": {
                "file_size": fingerprint.size,
                "file_name": fingerprint.filename,
                "keypoint_count": result["query_keypoint_count"],
                "minutiae_count": result.get("query_minutiae_count", 0),
                "ridge_features_count": result.get("query_ridge_features_count", 0)
            },
            "results": {
                "matches": result["matches"],
                "total_processed": result["total_processed"],
                "processing_time": result["processing_time"],
                "total_time": total_time,
                "analysis_techniques_used": {
                    "ridge_analysis": enable_ridge_analysis,
                    "orientation_analysis": enable_orientation_analysis,
                    "texture_analysis": enable_texture_analysis,
                    "minutiae_detection": enable_minutiae_detection
                }
            },
            "configuration": {
                "threshold": fingerprint_matcher.match_threshold,
                "batch_size": batch_size or 50
            },
            "officer_validation": officer_validation_result
        }
        logger.info(f"Advanced minutiae match response: {response}")
        return response

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error in advanced minutiae match endpoint: {error}")
        raise HTTPException(status_code=500, detail=str(error))

# @app.post("/sift-flann-match")
@app.post("/advanced-minutiae-match")
async def sift_flann_match(
    fingerprint: UploadFile = File(...),
    threshold: Optional[float] = Form(0.65),
    batch_size: Optional[int] = Form(50),
    max_results: Optional[int] = Form(10),
    enable_ridge_analysis: bool = Form(True),
    enable_orientation_analysis: bool = Form(True),
    enable_texture_analysis: bool = Form(True),
    enable_minutiae_detection: bool = Form(True)
):
    """SIFT + FLANN-based fingerprint matching over S3. Returns the best match by highest similarity."""
    try:
        if not fingerprint_matcher:
            raise HTTPException(status_code=500, detail="Fingerprint matcher not initialized")

        if fingerprint.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG, JPG, PNG, and TIFF are allowed."
            )

        start_time = datetime.now()
        file_content = await fingerprint.read()

        # Configure matcher threshold
        if threshold is not None:
            fingerprint_matcher.match_threshold = threshold

        # Persist uploaded content to a temp file and load with cv2.imread
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp.write(file_content)
            tmp.flush()
            query_path = tmp.name

        try:
            query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
            if query_img is None:
                raise HTTPException(status_code=400, detail="Failed to read uploaded image")

            sift = cv2.SIFT_create()
            query_kp, query_desc = sift.detectAndCompute(query_img, None)

            if query_desc is None or len(query_desc) == 0:
                raise HTTPException(status_code=400, detail="No SIFT descriptors found in query image")

            # Find matches across S3
            result = await sift_flann_find_matches(
                query_desc=query_desc,
                batch_size=batch_size or 50,
                sift=sift,
                ratio_thresh=0.75
            )

            # Sort and limit
            matches = sorted(result["matches"], key=lambda m: m.get("similarity", 0), reverse=True)
            if max_results and len(matches) > max_results:
                matches = matches[:max_results]

            total_time = (datetime.now() - start_time).total_seconds() * 1000

            best_result = matches[0] if matches else None
            response = {
                "success": True,
                "best_result": best_result
            }

            logger.info(f"SIFT-FLANN match response (best only): {response}")
            return response
        finally:
            try:
                os.unlink(query_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary query file: {e}")

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error in sift-flann-match endpoint: {error}")
        raise HTTPException(status_code=500, detail=str(error))

async def sift_flann_find_matches(
    query_desc: np.ndarray,
    batch_size: int,
    sift: Any,
    ratio_thresh: float = 0.75
) -> Dict[str, Any]:
    """Iterate S3 fingerprint images, compute SIFT+FLANN similarity to find top matches."""
    start_time = datetime.now()
    try:
        all_keys = await fingerprint_matcher.get_all_fingerprint_keys()
        if not all_keys:
            return {"matches": [], "total_processed": 0, "processing_time": 0.0}

        matches: List[Dict[str, Any]] = []
        total_processed = 0

        # Prepare FLANN matcher for SIFT (KDTree)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        async def process_key(key: str) -> Optional[Dict[str, Any]]:
            try:
                response = fingerprint_matcher.s3_client.get_object(Bucket=fingerprint_matcher.bucket_name, Key=key)
                image_buffer = response['Body'].read()

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    tmp.write(image_buffer)
                    tmp.flush()
                    target_path = tmp.name

                try:
                    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
                    if target_img is None:
                        return None

                    kp2, desc2 = sift.detectAndCompute(target_img, None)
                    if desc2 is None or len(desc2) == 0:
                        return None

                    # KNN match and Lowe's ratio test
                    raw_matches = flann.knnMatch(query_desc.astype(np.float32), desc2.astype(np.float32), k=2)
                    good = [m for m, n in raw_matches if m.distance < ratio_thresh * n.distance] if raw_matches else []

                    denom = float(min(len(query_desc), len(desc2))) or 1.0
                    similarity = float(len(good)) / denom

                    return {
                        "key": key,
                        "filename": key,
                        "similarity": similarity,
                        "good_matches": int(len(good)),
                        "target_keypoints": int(len(kp2) if kp2 else 0)
                    }
                finally:
                    try:
                        os.unlink(target_path)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Failed processing S3 key {key}: {e}")
                return None

        # Batched concurrent processing
        for i in range(0, len(all_keys), batch_size):
            batch = all_keys[i:i + batch_size]
            tasks = [process_key(k) for k in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, dict) and r is not None:
                    matches.append(r)
                total_processed += 1

        matches.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return {"matches": matches, "total_processed": total_processed, "processing_time": processing_time}
    except Exception as error:
        logger.error(f"Error in sift_flann_find_matches: {error}")
        raise

async def advanced_minutiae_find_matches(
    query_image_buffer: bytes, 
    batch_size: Optional[int] = None,
    enable_ridge_analysis: bool = True,
    enable_orientation_analysis: bool = True,
    enable_texture_analysis: bool = True,
    enable_minutiae_detection: bool = True
) -> Dict[str, Any]:
    """Advanced minutiae-based fingerprint matching with multiple analysis techniques"""
    try:
        start_time = datetime.now()
        
        # Save query image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(query_image_buffer)
            temp_file.flush()
            query_path = temp_file.name
        
        try:
            # Extract advanced query features
            query_features = await advanced_minutiae_extract_features(
                query_image_buffer,
                enable_ridge_analysis=enable_ridge_analysis,
                enable_orientation_analysis=enable_orientation_analysis,
                enable_texture_analysis=enable_texture_analysis,
                enable_minutiae_detection=enable_minutiae_detection
            )
            
            query_keypoint_count = len(query_features['keypoints']) if query_features and query_features.get('keypoints') else 0
            query_minutiae_count = len(query_features.get('minutiae_points', []))
            query_ridge_features_count = len(query_features.get('ridge_features', []))
            
            # Check if we have valid query features
            if not query_features:
                logger.error("Failed to extract features from query image")
                return {
                    "matches": [],
                    "total_processed": 0,
                    "processing_time": 0,
                    "query_keypoint_count": 0,
                    "query_minutiae_count": 0,
                    "query_ridge_features_count": 0
                }
            
            # Get all fingerprint keys from S3
            logger.info("Getting fingerprint keys from S3...")
            all_keys = await fingerprint_matcher.get_all_fingerprint_keys()
            
            # Handle None or empty results
            if all_keys is None or not all_keys:
                logger.warning("No fingerprint keys found in S3 bucket")
                return {
                    "matches": [],
                    "total_processed": 0,
                    "processing_time": 0,
                    "query_keypoint_count": query_keypoint_count,
                    "query_minutiae_count": query_minutiae_count,
                    "query_ridge_features_count": query_ridge_features_count
                }
            
            total_processed = 0
            matches = []
            
            # Set default batch size if not provided
            if batch_size is None:
                batch_size = 50
            
            # Process in batches
            for i in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[i:i + batch_size]
                
                # Process batch concurrently
                tasks = []
                for key in batch_keys:
                    task = advanced_minutiae_process_single_image(
                        key, 
                        query_path, 
                        query_features,
                        enable_ridge_analysis=enable_ridge_analysis,
                        enable_orientation_analysis=enable_orientation_analysis,
                        enable_texture_analysis=enable_texture_analysis,
                        enable_minutiae_detection=enable_minutiae_detection
                    )
                    tasks.append(task)
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect valid results
                for result in batch_results:
                    if isinstance(result, dict) and result.get('similarity', 0) >= fingerprint_matcher.match_threshold:
                        matches.append(result)
                    total_processed += 1
            
            # Sort matches by similarity score (descending)
            matches.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "matches": matches,
                "total_processed": total_processed,
                "processing_time": processing_time,
                "query_keypoint_count": query_keypoint_count,
                "query_minutiae_count": query_minutiae_count,
                "query_ridge_features_count": query_ridge_features_count
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(query_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary query file: {e}")
                
    except Exception as error:
        logger.error(f"Error in advanced_minutiae_find_matches: {error}")
        raise

async def advanced_minutiae_extract_features(
    image_buffer: bytes,
    enable_ridge_analysis: bool = True,
    enable_orientation_analysis: bool = True,
    enable_texture_analysis: bool = True,
    enable_minutiae_detection: bool = True
) -> Optional[Dict[str, Any]]:
    """Extract advanced minutiae features from image buffer"""
    try:
        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(image_buffer)
            temp_file.flush()
            temp_path = temp_file.name
        
        try:
            # Load image and extract features
            img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error("Failed to load image from buffer")
                return None
            
            # Preprocess image
            img = AdvancedFingerprintAnalysis._preprocess_image(img)
            
            # Extract advanced features
            features = {}
            
            # Basic ORB features
            orb = cv2.ORB_create(
                nfeatures=2000,
                scaleFactor=1.2,
                nlevels=12,
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                patchSize=31,
                fastThreshold=20
            )
            keypoints, descriptors = orb.detectAndCompute(img, None)
            features['keypoints'] = keypoints or []
            features['descriptors'] = descriptors
            
            # Advanced ridge analysis
            if enable_ridge_analysis:
                ridge_features = AdvancedFingerprintAnalysis._extract_ridge_features(img)
                features['ridge_features'] = ridge_features
            
            # Orientation field analysis
            if enable_orientation_analysis:
                orientation_field = AdvancedFingerprintAnalysis._extract_orientation_field(img)
                features['orientation_field'] = orientation_field
            
            # Texture analysis
            if enable_texture_analysis:
                texture_features = AdvancedFingerprintAnalysis._extract_texture_features(img)
                features['texture_features'] = texture_features
            
            # Minutiae point detection
            if enable_minutiae_detection:
                minutiae_points = AdvancedFingerprintAnalysis._detect_minutiae_points(img)
                features['minutiae_points'] = minutiae_points
            
            # Core and delta detection
            core_delta = AdvancedFingerprintAnalysis._detect_core_and_delta(img)
            features['core_delta'] = core_delta
            
            # Ridge count analysis
            ridge_count = AdvancedFingerprintAnalysis._analyze_ridge_count(img)
            features['ridge_count'] = ridge_count
            
            return features
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary feature extraction file: {e}")
                
    except Exception as error:
        logger.error(f"Error in advanced_minutiae_extract_features: {error}")
        return None

async def advanced_minutiae_process_single_image(
    key: str, 
    query_path: str, 
    query_features: Optional[Dict[str, Any]],
    enable_ridge_analysis: bool = True,
    enable_orientation_analysis: bool = True,
    enable_texture_analysis: bool = True,
    enable_minutiae_detection: bool = True
) -> Optional[Dict[str, Any]]:
    """Process a single image from S3 using advanced minutiae-based matching"""
    try:
        # Download image from S3
        try:
            response = fingerprint_matcher.s3_client.get_object(Bucket=fingerprint_matcher.bucket_name, Key=key)
            image_buffer = response['Body'].read()
        except Exception as e:
            logger.error(f"Failed to download image {key}: {e}")
            return None
        
        # Save S3 image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(image_buffer)
            temp_file.flush()
            s3_image_path = temp_file.name
        
        try:
            # Extract features from S3 image
            s3_features = await advanced_minutiae_extract_features(
                image_buffer,
                enable_ridge_analysis=enable_ridge_analysis,
                enable_orientation_analysis=enable_orientation_analysis,
                enable_texture_analysis=enable_texture_analysis,
                enable_minutiae_detection=enable_minutiae_detection
            )
            
            if not s3_features:
                return None
            
            # Calculate advanced similarity score
            similarity_score = AdvancedFingerprintAnalysis._calculate_advanced_similarity(
                query_features, 
                s3_features,
                enable_ridge_analysis=enable_ridge_analysis,
                enable_orientation_analysis=enable_orientation_analysis,
                enable_texture_analysis=enable_texture_analysis,
                enable_minutiae_detection=enable_minutiae_detection
            )
            logger.info(f"advanced-minutiae: key={key} similarity={float(similarity_score):.3f}")
            
            if similarity_score >= fingerprint_matcher.match_threshold:
                return {
                    "key": key,
                    "similarity": float(similarity_score),  # Convert to native Python float
                    "filename": key,
                    "size": len(image_buffer),
                    "match_quality": {
                        "excellent_match": bool(similarity_score >= 0.85),  # Convert to native Python bool
                        "very_good_match": bool(similarity_score >= 0.75),
                        "good_match": bool(similarity_score >= 0.65),
                        "fair_match": bool(similarity_score >= 0.55),
                        "poor_match": bool(similarity_score < 0.55)
                    },
                    "analysis_details": {
                        "keypoint_count": len(s3_features.get('keypoints', [])),
                        "minutiae_count": len(s3_features.get('minutiae_points', [])),
                        "ridge_features_count": len(s3_features.get('ridge_features', [])),
                        "has_core_delta": bool(s3_features.get('core_delta', {}))
                    }
                }
            
            return None
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(s3_image_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary S3 image file: {e}")
                
    except Exception as error:
        logger.error(f"Error processing image {key}: {error}")
        return None

class AdvancedFingerprintAnalysis:
    """Advanced fingerprint analysis techniques for highly accurate matching"""
    
    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:
        """Advanced preprocessing for fingerprint images"""
        # Resize to standard size
        image = cv2.resize(image, (512, 512))
        
        # Apply bilateral filter to reduce noise while preserving edges
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        # Apply morphological operations to enhance ridge structure
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Apply unsharp masking for edge enhancement
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        return image
    
    @staticmethod
    def _extract_ridge_features(image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract detailed ridge pattern features"""
        features = []
        
        # Apply Gabor filters at different orientations
        angles = [0, 45, 90, 135]
        frequencies = [0.1, 0.2, 0.3]
        
        for angle in angles:
            for freq in frequencies:
                # Create Gabor filter
                kernel = cv2.getGaborKernel((21, 21), 8, np.radians(angle), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                
                # Apply filter
                filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                
                # Extract features from filtered image
                for x in range(0, filtered.shape[1], 32):
                    for y in range(0, filtered.shape[0], 32):
                        if x < filtered.shape[1] and y < filtered.shape[0]:
                            region = filtered[y:min(y+32, filtered.shape[0]), x:min(x+32, filtered.shape[1])]
                            if region.size > 0:
                                features.append({
                                    'x': int(x), 'y': int(y),
                                    'angle': int(angle),
                                    'frequency': float(freq),
                                    'mean_intensity': float(np.mean(region)),
                                    'std_intensity': float(np.std(region)),
                                    'max_intensity': float(np.max(region)),
                                    'min_intensity': float(np.min(region))
                                })
        
        return features[:200]  # Limit to 200 features
    
    @staticmethod
    def _extract_orientation_field(image: np.ndarray) -> np.ndarray:
        """Extract orientation field from fingerprint"""
        # Calculate gradients
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate orientation
        orientation = np.arctan2(sobel_y, sobel_x)
        
        # Smooth orientation field
        orientation = cv2.GaussianBlur(orientation, (5, 5), 0)
        
        # Convert to native Python types for JSON serialization
        return orientation.astype(np.float64)
    
    @staticmethod
    def _extract_texture_features(image: np.ndarray) -> List[float]:
        """Extract texture features using Local Binary Patterns"""
        features = []
        
        # Calculate LBP for different scales
        scales = [1, 2, 3]
        
        for scale in scales:
            # Resize image
            new_width = max(1, image.shape[1] // scale)
            new_height = max(1, image.shape[0] // scale)
            scaled_img = cv2.resize(image, (new_width, new_height))
            
            # Calculate LBP
            lbp = AdvancedFingerprintAnalysis._calculate_lbp(scaled_img)
            
            # Calculate histogram
            hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            hist = hist.astype(float) / hist.sum()  # Normalize
            
            features.extend([float(x) for x in hist.tolist()])
        
        return features[:100]  # Limit to 100 features
    
    @staticmethod
    def _calculate_lbp(image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern"""
        lbp = np.zeros_like(image)
        
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                center = image[i, j]
                code = 0
                
                # 8-neighbor LBP
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    @staticmethod
    def _detect_minutiae_points(image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect minutiae points (ridge endings and bifurcations)"""
        minutiae = []
        
        # Apply ridge detection
        try:
            ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
            ridges = ridge_filter.getRidgeFilteredImage(image)
        except AttributeError:
            # Fallback to edge detection
            ridges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(ridges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Analyze contour for minutiae characteristics
            if len(contour) > 5:  # Minimum contour length
                # Calculate contour properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if area > 10 and perimeter > 20:  # Filter small contours
                    # Get contour center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Determine minutiae type based on contour properties
                        if perimeter / area > 0.5:  # Thin, long contour - likely ridge ending
                            minutiae.append({
                                'x': int(cx), 'y': int(cy),
                                'type': 'ending',
                                'area': float(area),
                                'perimeter': float(perimeter)
                            })
                        else:  # Thicker contour - likely bifurcation
                            minutiae.append({
                                'x': int(cx), 'y': int(cy),
                                'type': 'bifurcation',
                                'area': float(area),
                                'perimeter': float(perimeter)
                            })
        
        return minutiae[:50]  # Limit to 50 minutiae points
    
    @staticmethod
    def _detect_core_and_delta(image: np.ndarray) -> Dict[str, Any]:
        """Detect core and delta points in fingerprint"""
        result = {}
        
        # Apply orientation field analysis
        orientation = AdvancedFingerprintAnalysis._extract_orientation_field(image)
        
        # Find regions with high orientation variance (potential core/delta)
        orientation_variance = cv2.Laplacian(orientation, cv2.CV_64F)
        
        # Find local maxima for core points
        core_points = []
        delta_points = []
        
        # Simple heuristic: high variance regions near center are cores
        # High variance regions near edges are deltas
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        
        for y in range(0, image.shape[0], 16):
            for x in range(0, image.shape[1], 16):
                if x < image.shape[1] and y < image.shape[0]:
                    variance = abs(orientation_variance[y, x])
                    
                    if variance > 0.5:  # Threshold for significant orientation change
                        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        
                        if distance_from_center < 100:  # Near center - likely core
                            core_points.append({'x': int(x), 'y': int(y), 'variance': float(variance)})
                        else:  # Near edge - likely delta
                            delta_points.append({'x': int(x), 'y': int(y), 'variance': float(variance)})
        
        result['core_points'] = core_points[:3]  # Top 3 core candidates
        result['delta_points'] = delta_points[:3]  # Top 3 delta candidates
        
        return result
    
    @staticmethod
    def _analyze_ridge_count(image: np.ndarray) -> Dict[str, Any]:
        """Analyze ridge count patterns"""
        # Apply ridge detection
        try:
            ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
            ridges = ridge_filter.getRidgeFilteredImage(image)
        except AttributeError:
            ridges = cv2.Canny(image, 50, 150)
        
        # Count ridges in different regions
        regions = []
        region_size = 64
        
        for y in range(0, image.shape[0], region_size):
            for x in range(0, image.shape[1], region_size):
                if x + region_size <= image.shape[1] and y + region_size <= image.shape[0]:
                    region = ridges[y:y+region_size, x:x+region_size]
                    ridge_count = np.sum(region > 0)
                    regions.append({
                        'x': int(x), 'y': int(y),
                        'ridge_count': int(ridge_count),
                        'density': float(ridge_count / (region_size * region_size))
                    })
        
        return {
            'regions': regions,
            'total_ridges': int(sum(r['ridge_count'] for r in regions)),
            'avg_density': float(np.mean([r['density'] for r in regions]))
        }
    
    @staticmethod
    def _calculate_advanced_similarity(
        features1: Dict[str, Any], 
        features2: Dict[str, Any],
        enable_ridge_analysis: bool = True,
        enable_orientation_analysis: bool = True,
        enable_texture_analysis: bool = True,
        enable_minutiae_detection: bool = True
    ) -> float:
        """Calculate advanced similarity score using multiple techniques"""
        try:
            similarities = []
            weights = []
            
            # ORB descriptor similarity
            if features1.get('descriptors') is not None and features2.get('descriptors') is not None:
                orb_similarity = AdvancedFingerprintAnalysis._calculate_orb_similarity(
                    features1['descriptors'], features2['descriptors']
                )
                similarities.append(orb_similarity)
                weights.append(0.25)
            
            # Ridge pattern similarity
            if enable_ridge_analysis and features1.get('ridge_features') and features2.get('ridge_features'):
                ridge_similarity = AdvancedFingerprintAnalysis._calculate_ridge_similarity(
                    features1['ridge_features'], features2['ridge_features']
                )
                similarities.append(ridge_similarity)
                weights.append(0.20)
            
            # Orientation field similarity
            if enable_orientation_analysis and features1.get('orientation_field') is not None and features2.get('orientation_field') is not None:
                orientation_similarity = AdvancedFingerprintAnalysis._calculate_orientation_similarity(
                    features1['orientation_field'], features2['orientation_field']
                )
                similarities.append(orientation_similarity)
                weights.append(0.20)
            
            # Texture similarity
            if enable_texture_analysis and features1.get('texture_features') and features2.get('texture_features'):
                texture_similarity = AdvancedFingerprintAnalysis._calculate_texture_similarity(
                    features1['texture_features'], features2['texture_features']
                )
                similarities.append(texture_similarity)
                weights.append(0.15)
            
            # Minutiae similarity
            if enable_minutiae_detection and features1.get('minutiae_points') and features2.get('minutiae_points'):
                minutiae_similarity = AdvancedFingerprintAnalysis._calculate_minutiae_similarity(
                    features1['minutiae_points'], features2['minutiae_points']
                )
                similarities.append(minutiae_similarity)
                weights.append(0.20)
            
            # If no similarities calculated, return 0
            if not similarities:
                return 0.0
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            # Calculate weighted average and convert to native Python float
            final_similarity = float(sum(s * w for s, w in zip(similarities, weights)))
            
            return min(1.0, max(0.0, final_similarity))
            
        except Exception as e:
            logger.error(f"Error calculating advanced similarity: {e}")
            return 0.0
    
    @staticmethod
    def _calculate_orb_similarity(desc1, desc2) -> float:
        """Calculate ORB descriptor similarity"""
        try:
            if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
                return 0.0
            
            # Use brute force matcher for binary descriptors
            bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf_matcher.match(desc1, desc2)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Take top matches
            max_matches = min(50, len(matches))
            good_matches = matches[:max_matches]
            
            # Calculate similarity based on number of good matches
            max_possible_matches = min(len(desc1), len(desc2))
            if max_possible_matches == 0:
                return 0.0
            
            similarity = float(len(good_matches) / max_possible_matches)
            return min(similarity, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating ORB similarity: {e}")
            return 0.0
    
    @staticmethod
    def _calculate_ridge_similarity(ridges1: List[Dict], ridges2: List[Dict]) -> float:
        """Calculate ridge pattern similarity"""
        try:
            if not ridges1 or not ridges2:
                return 0.0
            
            # Compare ridge features using correlation
            features1 = []
            features2 = []
            
            # Extract numerical features for comparison
            for ridge in ridges1:
                features1.extend([
                    ridge.get('mean_intensity', 0),
                    ridge.get('std_intensity', 0),
                    ridge.get('max_intensity', 0),
                    ridge.get('min_intensity', 0)
                ])
            
            for ridge in ridges2:
                features2.extend([
                    ridge.get('mean_intensity', 0),
                    ridge.get('std_intensity', 0),
                    ridge.get('max_intensity', 0),
                    ridge.get('min_intensity', 0)
                ])
            
            # Pad shorter array
            max_len = max(len(features1), len(features2))
            features1.extend([0] * (max_len - len(features1)))
            features2.extend([0] * (max_len - len(features2)))
            
            # Calculate correlation and convert to native Python float
            correlation = float(np.corrcoef(features1, features2)[0, 1])
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating ridge similarity: {e}")
            return 0.0
    
    @staticmethod
    def _calculate_orientation_similarity(orient1: np.ndarray, orient2: np.ndarray) -> float:
        """Calculate orientation field similarity"""
        try:
            if orient1 is None or orient2 is None:
                return 0.0
            
            # Resize to same size if different
            if orient1.shape != orient2.shape:
                orient2 = cv2.resize(orient2, (orient1.shape[1], orient1.shape[0]))
            
            # Calculate difference
            diff = np.abs(orient1 - orient2)
            
            # Normalize difference and convert to native Python float
            similarity = float(1.0 - np.mean(diff) / np.pi)
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"Error calculating orientation similarity: {e}")
            return 0.0
    
    @staticmethod
    def _calculate_texture_similarity(texture1: List[float], texture2: List[float]) -> float:
        """Calculate texture similarity"""
        try:
            if not texture1 or not texture2:
                return 0.0
            
            # Pad shorter array
            max_len = max(len(texture1), len(texture2))
            texture1.extend([0] * (max_len - len(texture1)))
            texture2.extend([0] * (max_len - len(texture2)))
            
            # Calculate correlation and convert to native Python float
            correlation = float(np.corrcoef(texture1, texture2)[0, 1])
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating texture similarity: {e}")
            return 0.0
    
    @staticmethod
    def _calculate_minutiae_similarity(minutiae1: List[Dict], minutiae2: List[Dict]) -> float:
        """Calculate minutiae point similarity"""
        try:
            if not minutiae1 or not minutiae2:
                return 0.0
            
            # Compare minutiae positions and types
            matches = 0
            total_comparisons = 0
            
            for m1 in minutiae1:
                for m2 in minutiae2:
                    # Calculate distance between minutiae
                    distance = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
                    
                    # Consider it a match if close and same type
                    if distance < 20 and m1.get('type') == m2.get('type'):
                        matches += 1
                    
                    total_comparisons += 1
            
            if total_comparisons == 0:
                return 0.0
            
            similarity = float(matches / total_comparisons)
            return min(similarity, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating minutiae similarity: {e}")
            return 0.0

class MinutiaeVerification:
    """High-accuracy minutiae extraction and verification between two fingerprints."""

    @staticmethod
    def _preprocess_for_skeleton(image: np.ndarray) -> np.ndarray:
        """Preprocess image to enhance ridges and prepare for skeletonization."""
        image = cv2.resize(image, (512, 512))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = cv2.bilateralFilter(image, 7, 50, 50)
        try:
            thresh = threshold_local(image, block_size=31, offset=5)
            binary = (image > thresh).astype(np.uint8) * 255
        except Exception:
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary) > 127:
            binary = 255 - binary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    @staticmethod
    def _compute_orientation_field(image_gray: np.ndarray) -> np.ndarray:
        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        orientation = np.arctan2(sobel_y, sobel_x)
        orientation = cv2.GaussianBlur(orientation, (7, 7), 0)
        return orientation

    @staticmethod
    def _extract_minutiae_points(binary_image: np.ndarray, orientation_field: np.ndarray) -> List[Dict[str, Any]]:
        skel = skeletonize((binary_image > 0)).astype(np.uint8)
        skel_padded = np.pad(skel, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        minutiae: List[Dict[str, Any]] = []
        height, width = skel.shape
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if skel[y, x] == 0:
                    continue
                nb = 0
                for dy, dx in neighbors:
                    if skel_padded[y + 1 + dy, x + 1 + dx] > 0:
                        nb += 1
                if nb == 1:
                    m_type = 'ending'
                elif nb == 3:
                    m_type = 'bifurcation'
                else:
                    continue
                angle = float(orientation_field[y, x])
                minutiae.append({'x': int(x), 'y': int(y), 'type': m_type, 'angle': angle})
        return MinutiaeVerification._suppress_close_minutiae(minutiae, min_distance=8)

    @staticmethod
    def _suppress_close_minutiae(minutiae: List[Dict[str, Any]], min_distance: int = 8) -> List[Dict[str, Any]]:
        if not minutiae:
            return []
        points = np.array([[m['x'], m['y']] for m in minutiae], dtype=np.float32)
        kept = []
        taken = np.zeros(len(points), dtype=bool)
        tree = cKDTree(points)
        for i in range(len(points)):
            if taken[i]:
                continue
            idxs = tree.query_ball_point(points[i], r=min_distance)
            candidates = [minutiae[j] for j in idxs]
            chosen = None
            for c in candidates:
                if c['type'] == 'bifurcation':
                    chosen = c
                    break
            if chosen is None:
                chosen = candidates[0]
            kept.append(chosen)
            for j in idxs:
                taken[j] = True
        return kept

    @staticmethod
    def _build_descriptors(minutiae: List[Dict[str, Any]]) -> np.ndarray:
        if not minutiae:
            return np.empty((0, 3), dtype=np.float32)
        angles = np.array([m['angle'] for m in minutiae], dtype=np.float32)
        types = np.array([1.0 if m['type'] == 'bifurcation' else 0.0 for m in minutiae], dtype=np.float32)
        descriptors = np.stack([np.cos(angles), np.sin(angles), types], axis=1)
        return descriptors

    @staticmethod
    def _estimate_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if len(src_pts) < 3 or len(dst_pts) < 3:
            return None, None
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=10.0, maxIters=2000, confidence=0.99)
        return M, inliers

    @staticmethod
    def _score_match(M: Optional[np.ndarray], inliers_mask: Optional[np.ndarray], total_minutiae: int) -> float:
        if M is None or inliers_mask is None or total_minutiae == 0:
            return 0.0
        inliers = int(np.sum(inliers_mask)) if inliers_mask is not None else 0
        score = float(inliers / max(1, total_minutiae))
        return min(1.0, max(0.0, score))

    @staticmethod
    def verify(image1: np.ndarray, image2: np.ndarray, min_inliers_required: int = 12) -> Dict[str, Any]:
        bin1 = MinutiaeVerification._preprocess_for_skeleton(image1)
        bin2 = MinutiaeVerification._preprocess_for_skeleton(image2)
        ori1 = MinutiaeVerification._compute_orientation_field(image1)
        ori2 = MinutiaeVerification._compute_orientation_field(image2)
        minutiae1 = MinutiaeVerification._extract_minutiae_points(bin1, ori1)
        minutiae2 = MinutiaeVerification._extract_minutiae_points(bin2, ori2)
        desc1 = MinutiaeVerification._build_descriptors(minutiae1)
        desc2 = MinutiaeVerification._build_descriptors(minutiae2)
        pts1 = np.array([[m['x'], m['y']] for m in minutiae1], dtype=np.float32)
        pts2 = np.array([[m['x'], m['y']] for m in minutiae2], dtype=np.float32)
        if len(pts1) < 8 or len(pts2) < 8:
            return {'score': 0.0, 'inliers': 0, 'transform': None, 'minutiae_counts': {'image1': len(pts1), 'image2': len(pts2)}, 'is_match': False}
        correspondences_src = []
        correspondences_dst = []
        if len(desc2) > 0:
            tree = cKDTree(desc2)
            for i, d in enumerate(desc1):
                dist, j = tree.query(d, k=1)
                if np.isfinite(dist) and dist < 0.75:
                    correspondences_src.append(pts1[i])
                    correspondences_dst.append(pts2[j])
        if len(correspondences_src) < 8:
            tree_pts = cKDTree(pts2)
            for p in pts1:
                dist, j = tree_pts.query(p, k=1)
                if np.isfinite(dist) and dist < 40.0:
                    correspondences_src.append(p)
                    correspondences_dst.append(pts2[j])
        if len(correspondences_src) < 3:
            return {'score': 0.0, 'inliers': 0, 'transform': None, 'minutiae_counts': {'image1': len(pts1), 'image2': len(pts2)}, 'is_match': False}
        src = np.vstack(correspondences_src).astype(np.float32)
        dst = np.vstack(correspondences_dst).astype(np.float32)
        M, inliers_mask = MinutiaeVerification._estimate_similarity_transform(src, dst)
        total_minutiae = min(len(pts1), len(pts2))
        score = MinutiaeVerification._score_match(M, inliers_mask, total_minutiae)
        inliers = int(np.sum(inliers_mask)) if inliers_mask is not None else 0
        transform_details = None
        if M is not None:
            a, b = M[0, 0], M[0, 1]
            scale = float(np.sqrt(a * a + b * b))
            rotation = float(np.degrees(np.arctan2(b, a)))
            tx, ty = float(M[0, 2]), float(M[1, 2])
            transform_details = {'scale': scale, 'rotation_deg': rotation, 'translation': [tx, ty]}
        return {
            'score': float(score),
            'inliers': int(inliers),
            'transform': transform_details,
            'minutiae_counts': {'image1': int(len(pts1)), 'image2': int(len(pts2))},
            'is_match': bool(inliers >= min_inliers_required and score >= 0.5)
        }

    @staticmethod
    def extract_features(image: np.ndarray) -> Dict[str, Any]:
        """Extract reusable minutiae features for efficiency across multiple comparisons."""
        image = cv2.resize(image, (512, 512))
        bin_img = MinutiaeVerification._preprocess_for_skeleton(image)
        ori = MinutiaeVerification._compute_orientation_field(image)
        minutiae = MinutiaeVerification._extract_minutiae_points(bin_img, ori)
        desc = MinutiaeVerification._build_descriptors(minutiae)
        pts = np.array([[m['x'], m['y']] for m in minutiae], dtype=np.float32)
        return {
            'pts': pts,
            'desc': desc,
            'minutiae': minutiae,
            'counts': {
                'minutiae': int(len(minutiae))
            }
        }

    @staticmethod
    def compare_features(f1: Dict[str, Any], f2: Dict[str, Any], min_inliers_required: int = 12) -> Dict[str, Any]:
        """Compare two precomputed minutiae feature sets and return match details."""
        pts1: np.ndarray = f1.get('pts', np.empty((0, 2), dtype=np.float32))
        pts2: np.ndarray = f2.get('pts', np.empty((0, 2), dtype=np.float32))
        desc1: np.ndarray = f1.get('desc', np.empty((0, 3), dtype=np.float32))
        desc2: np.ndarray = f2.get('desc', np.empty((0, 3), dtype=np.float32))

        if len(pts1) < 8 or len(pts2) < 8:
            return {'score': 0.0, 'inliers': 0, 'transform': None, 'is_match': False}

        correspondences_src = []
        correspondences_dst = []

        if len(desc2) > 0:
            tree = cKDTree(desc2)
            for i, d in enumerate(desc1):
                dist, j = tree.query(d, k=1)
                if np.isfinite(dist) and dist < 0.75:
                    correspondences_src.append(pts1[i])
                    correspondences_dst.append(pts2[j])

        if len(correspondences_src) < 8:
            tree_pts = cKDTree(pts2)
            for p in pts1:
                dist, j = tree_pts.query(p, k=1)
                if np.isfinite(dist) and dist < 40.0:
                    correspondences_src.append(p)
                    correspondences_dst.append(pts2[j])

        if len(correspondences_src) < 3:
            return {'score': 0.0, 'inliers': 0, 'transform': None, 'is_match': False}

        src = np.vstack(correspondences_src).astype(np.float32)
        dst = np.vstack(correspondences_dst).astype(np.float32)

        M, inliers_mask = MinutiaeVerification._estimate_similarity_transform(src, dst)
        total_minutiae = min(len(pts1), len(pts2))
        score = MinutiaeVerification._score_match(M, inliers_mask, total_minutiae)
        inliers = int(np.sum(inliers_mask)) if inliers_mask is not None else 0

        transform_details = None
        if M is not None:
            a, b = M[0, 0], M[0, 1]
            scale = float(np.sqrt(a * a + b * b))
            rotation = float(np.degrees(np.arctan2(b, a)))
            tx, ty = float(M[0, 2]), float(M[1, 2])
            transform_details = {'scale': scale, 'rotation_deg': rotation, 'translation': [tx, ty]}

        return {
            'score': float(score),
            'inliers': int(inliers),
            'transform': transform_details,
            'is_match': bool(inliers >= min_inliers_required and score >= 0.5)
        }

class MCCMatcher:
    """Minutia Cylinder-Code inspired matcher for robust and accurate fingerprint matching."""

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        two_pi = 2 * np.pi
        angle = angle % two_pi
        if angle < 0:
            angle += two_pi
        return angle

    @staticmethod
    def _build_mcc_descriptors(minutiae: List[Dict[str, Any]], num_sectors: int = 16, num_rings: int = 5, radius: float = 60.0) -> np.ndarray:
        if not minutiae:
            return np.empty((0, num_rings * num_sectors), dtype=np.float32)

        points = np.array([[m['x'], m['y']] for m in minutiae], dtype=np.float32)
        angles = np.array([m.get('angle', 0.0) for m in minutiae], dtype=np.float32)
        types = np.array([1 if m.get('type') == 'bifurcation' else 0 for m in minutiae], dtype=np.int32)
        tree = cKDTree(points)

        descriptors = np.zeros((len(minutiae), num_rings * num_sectors), dtype=np.float32)
        ring_edges = np.linspace(0.0, radius, num_rings + 1)
        sector_width = 2 * np.pi / num_sectors

        for i in range(len(minutiae)):
            center = points[i]
            center_angle = float(angles[i])
            center_type = types[i]

            # Get neighbors within radius
            idxs = tree.query_ball_point(center, r=radius)
            if i in idxs:
                idxs.remove(i)
            if not idxs:
                continue

            hist = np.zeros((num_rings, num_sectors), dtype=np.float32)
            for j in idxs:
                vec = points[j] - center
                dist = float(np.hypot(vec[0], vec[1]))
                if dist <= 0.0 or dist > radius:
                    continue
                ring_idx = int(np.searchsorted(ring_edges, dist, side='right') - 1)
                ring_idx = max(0, min(num_rings - 1, ring_idx))

                rel_angle = np.arctan2(vec[1], vec[0]) - center_angle
                rel_angle = MCCMatcher._wrap_angle(float(rel_angle))
                sector_idx = int(rel_angle / sector_width)
                sector_idx = max(0, min(num_sectors - 1, sector_idx))

                # Weight by orientation consistency and same-type preference
                neighbor_angle = float(angles[j])
                ori_diff = abs(MCCMatcher._wrap_angle(neighbor_angle - center_angle))
                ori_weight = float(np.cos(ori_diff))
                ori_weight = max(0.0, ori_weight)  # penalize opposite directions
                type_weight = 1.2 if types[j] == center_type else 0.9
                radial_weight = 1.0 - (dist / radius) * 0.3  # slight preference for closer neighbors

                weight = float(1.0 * type_weight * (0.5 + 0.5 * ori_weight) * radial_weight)
                hist[ring_idx, sector_idx] += weight

            # Normalize histogram
            vec = hist.reshape(-1)
            norm = np.linalg.norm(vec) + 1e-8
            descriptors[i] = vec / norm

        return descriptors

    @staticmethod
    def extract_features(image: np.ndarray, num_sectors: int = 16, num_rings: int = 5, radius: float = 60.0) -> Dict[str, Any]:
        # Reuse minutiae extraction from MinutiaeVerification
        image = cv2.resize(image, (512, 512))
        bin_img = MinutiaeVerification._preprocess_for_skeleton(image)
        ori = MinutiaeVerification._compute_orientation_field(image)
        minutiae = MinutiaeVerification._extract_minutiae_points(bin_img, ori)
        pts = np.array([[m['x'], m['y']] for m in minutiae], dtype=np.float32)
        descs = MCCMatcher._build_mcc_descriptors(minutiae, num_sectors=num_sectors, num_rings=num_rings, radius=radius)
        return {
            'minutiae': minutiae,
            'pts': pts,
            'descs': descs
        }

    @staticmethod
    def compare_features(f1: Dict[str, Any], f2: Dict[str, Any], min_inliers_required: int = 12) -> Dict[str, Any]:
        pts1: np.ndarray = f1.get('pts', np.empty((0, 2), dtype=np.float32))
        pts2: np.ndarray = f2.get('pts', np.empty((0, 2), dtype=np.float32))
        d1: np.ndarray = f1.get('descs', np.empty((0, 1), dtype=np.float32))
        d2: np.ndarray = f2.get('descs', np.empty((0, 1), dtype=np.float32))
        if len(pts1) < 8 or len(pts2) < 8 or len(d1) == 0 or len(d2) == 0:
            return {'similarity': 0.0, 'inliers': 0, 'transform': None}

        # KDTree for descriptor NN matching (Euclidean)
        tree = cKDTree(d2)
        distances, indices = tree.query(d1, k=2, p=2, workers=-1)

        correspondences_src = []
        correspondences_dst = []
        desc_dists = []

        for i in range(len(d1)):
            dists = distances[i]
            idxs = indices[i]
            if not np.all(np.isfinite(dists)):
                continue
            # Lowe ratio test
            if len(dists) == 2 and dists[0] < 0.85 * dists[1]:
                correspondences_src.append(pts1[i])
                correspondences_dst.append(pts2[idxs[0]])
                desc_dists.append(dists[0])

        if len(correspondences_src) < 4:
            return {'similarity': 0.0, 'inliers': 0, 'transform': None}

        src = np.vstack(correspondences_src).astype(np.float32)
        dst = np.vstack(correspondences_dst).astype(np.float32)
        M, inliers_mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0, maxIters=3000, confidence=0.995)
        if M is None or inliers_mask is None:
            return {'similarity': 0.0, 'inliers': 0, 'transform': None}

        inliers = int(np.sum(inliers_mask))
        if inliers <= 0:
            return {'similarity': 0.0, 'inliers': 0, 'transform': None}

        # Similarity: weighted by inlier ratio and descriptor quality
        min_total = min(len(pts1), len(pts2))
        inlier_ratio = float(inliers / max(1, min_total))
        med_desc = float(np.median(desc_dists)) if desc_dists else 1.0
        desc_score = float(np.exp(-med_desc))  # smaller distances -> higher score
        similarity = 0.75 * inlier_ratio + 0.25 * desc_score
        similarity = float(max(0.0, min(1.0, similarity)))

        a, b = M[0, 0], M[0, 1]
        scale = float(np.sqrt(a * a + b * b))
        rotation = float(np.degrees(np.arctan2(b, a)))
        tx, ty = float(M[0, 2]), float(M[1, 2])
        transform_details = {'scale': scale, 'rotation_deg': rotation, 'translation': [tx, ty]}

        return {
            'similarity': similarity,
            'inliers': inliers,
            'transform': transform_details
        }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "GET /health",
                "GET /info",
                "POST /match",
                "POST /match/base64",
                "POST /minutiaematch",
                "POST /advanced-minutiae-match",
                "POST /comprehensive-minutiae-match",
                "POST /upload-fingerprint",
                "POST /ultra-minutiae-match",
                "POST /mcc-minutiae-match",
                "POST /nbis-match",
                "POST /nbis-verify",
                "POST /verify-minutiae",
                "GET /cache/stats",
                "POST /cache/clear"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.post("/comprehensive-minutiae-match")
async def comprehensive_minutiae_match(
    fingerprint: UploadFile = File(...),
    threshold: Optional[float] = Form(0.4),
    batch_size: Optional[int] = Form(50),
    max_results: Optional[int] = Form(10),
    enable_minutiae_matching: bool = Form(True),
    enable_sift_matching: bool = Form(True),
    dist_threshold: Optional[int] = Form(15),
    angle_threshold: Optional[int] = Form(20),
    ratio_threshold: Optional[float] = Form(0.75)
):
    """Comprehensive fingerprint matching using MinutiaeFeature, SIFT, and AWS S3 integration"""
    try:
        if not fingerprint_matcher:
            raise HTTPException(status_code=500, detail="Fingerprint matcher not initialized")

        # Validate file type
        if fingerprint.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG, JPG, PNG, and TIFF are allowed."
            )

        start_time = datetime.now()
        logger.info(f"Processing comprehensive minutiae match request - File size: {fingerprint.size} bytes")

        # Read file content
        file_content = await fingerprint.read()

        # Save query image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            query_path = temp_file.name

        try:
            # Load query image
            query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
            if query_img is None:
                raise HTTPException(status_code=400, detail="Failed to load query image")

            # Initialize extractors and matchers
            minutiae_extractor = MinutiaeExtractor()
            minutiae_matcher = MinutiaeMatcher(
                dist_thresh=dist_threshold,
                angle_thresh=angle_threshold
            )
            sift_matcher = SIFTMatcher(ratio_thresh=ratio_threshold)

            # Extract features from query image
            query_term, query_bif = minutiae_extractor.extract(query_img)
            query_minutiae_count = len(query_term) + len(query_bif)

            logger.info(f"Extracted {len(query_term)} termination and {len(query_bif)} bifurcation features from query image")

            # Get all fingerprint keys from S3
            logger.info("Getting fingerprint keys from S3...")
            all_keys = await fingerprint_matcher.get_all_fingerprint_keys()
            
            if all_keys is None or not all_keys:
                logger.warning("No fingerprint keys found in S3 bucket")
                response = {
                    "success": True,
                    "query": {
                        "file_size": fingerprint.size,
                        "file_name": fingerprint.filename,
                        "minutiae_count": query_minutiae_count
                    },
                    "results": {
                        "matches": [],
                        "total_processed": 0,
                        "processing_time": 0,
                        "total_time": 0
                    },
                    "configuration": {
                        "threshold": threshold,
                        "batch_size": batch_size or 50,
                        "enable_minutiae_matching": enable_minutiae_matching,
                        "enable_sift_matching": enable_sift_matching,
                        "dist_threshold": dist_threshold,
                        "angle_threshold": angle_threshold,
                        "ratio_threshold": ratio_threshold
                    }
                }
                logger.info(f"Comprehensive minutiae match response: {response}")
                return response

            total_processed = 0
            matches = []

            # Set default batch size if not provided
            if batch_size is None:
                batch_size = 50

            # Process in batches
            for i in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[i:i + batch_size]
                
                # Process batch concurrently
                tasks = []
                for key in batch_keys:
                    task = comprehensive_process_single_image(
                        key, 
                        query_img, 
                        query_term, 
                        query_bif,
                        minutiae_extractor,
                        minutiae_matcher,
                        sift_matcher,
                        enable_minutiae_matching,
                        enable_sift_matching,
                        threshold
                    )
                    tasks.append(task)
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect valid results
                for result in batch_results:
                    if isinstance(result, dict) and result.get('similarity_score', 0) >= threshold:
                        matches.append(result)
                    total_processed += 1

            # Sort matches by similarity score (descending)
            matches.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

            # Limit results if specified
            if max_results and len(matches) > max_results:
                matches = matches[:max_results]

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            response = {
                "success": True,
                "query": {
                    "file_size": fingerprint.size,
                    "file_name": fingerprint.filename,
                    "minutiae_count": query_minutiae_count,
                    "termination_count": len(query_term),
                    "bifurcation_count": len(query_bif)
                },
                "results": {
                    "matches": matches,
                    "total_processed": total_processed,
                    "processing_time": processing_time,
                    "total_time": processing_time
                },
                "configuration": {
                    "threshold": threshold,
                    "batch_size": batch_size,
                    "enable_minutiae_matching": enable_minutiae_matching,
                    "enable_sift_matching": enable_sift_matching,
                    "dist_threshold": dist_threshold,
                    "angle_threshold": angle_threshold,
                    "ratio_threshold": ratio_threshold
                }
            }
            logger.info(f"Comprehensive minutiae match response: {response}")
            return response

        finally:
            # Clean up temporary file
            try:
                os.unlink(query_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary query file: {e}")

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error in comprehensive minutiae match endpoint: {error}")
        raise HTTPException(status_code=500, detail=str(error))

async def comprehensive_process_single_image(
    key: str,
    query_img: np.ndarray,
    query_term: List,
    query_bif: List,
    minutiae_extractor: MinutiaeExtractor,
    minutiae_matcher: MinutiaeMatcher,
    sift_matcher: SIFTMatcher,
    enable_minutiae_matching: bool,
    enable_sift_matching: bool,
    threshold: float
) -> Optional[Dict[str, Any]]:
    """Process a single image from S3 using comprehensive minutiae and SIFT matching"""
    try:
        # Download image from S3
        try:
            response = fingerprint_matcher.s3_client.get_object(Bucket=fingerprint_matcher.bucket_name, Key=key)
            image_buffer = response['Body'].read()
        except Exception as e:
            logger.error(f"Failed to download image {key}: {e}")
            return None

        # Save S3 image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(image_buffer)
            temp_file.flush()
            s3_image_path = temp_file.name

        try:
            # Load target image
            target_img = cv2.imread(s3_image_path, cv2.IMREAD_GRAYSCALE)
            if target_img is None:
                return None

            # Extract features from target image
            target_term, target_bif = minutiae_extractor.extract(target_img)
            target_minutiae_count = len(target_term) + len(target_bif)

            # Calculate similarity scores
            score_minutiae = 0.0
            score_sift = 0.0

            if enable_minutiae_matching:
                score_minutiae = minutiae_matcher.match(query_term, target_term)
                logger.debug(f"Minutiae score for {key}: {score_minutiae:.3f}")

            if enable_sift_matching:
                score_sift = sift_matcher.match(query_img, target_img)
                logger.debug(f"SIFT score for {key}: {score_sift:.3f}")

            # Calculate final score
            if enable_minutiae_matching and enable_sift_matching:
                final_score = max(score_minutiae, score_sift)
            elif enable_minutiae_matching:
                final_score = score_minutiae
            elif enable_sift_matching:
                final_score = score_sift
            else:
                final_score = 0.0

            logger.info(f"comprehensive: key={key} minutiae={float(score_minutiae):.3f} sift={float(score_sift):.3f} final={float(final_score):.3f}")

            if final_score >= threshold:
                return {
                    "s3_key": key,
                    "filename": key,
                    "similarity_score": round(final_score, 3),
                    "minutiae_score": round(score_minutiae, 3) if enable_minutiae_matching else None,
                    "sift_score": round(score_sift, 3) if enable_sift_matching else None,
                    "size": len(image_buffer),
                    "target_minutiae_count": target_minutiae_count,
                    "target_termination_count": len(target_term),
                    "target_bifurcation_count": len(target_bif),
                    "match_quality": {
                        "excellent_match": final_score >= 0.8,
                        "very_good_match": final_score >= 0.6,
                        "good_match": final_score >= 0.4,
                        "fair_match": final_score >= 0.2,
                        "poor_match": final_score < 0.2
                    }
                }

            return None

        finally:
            # Clean up temporary file
            try:
                os.unlink(s3_image_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary S3 image file: {e}")

    except Exception as error:
        logger.error(f"Error processing image {key}: {error}")
        return None

@app.post("/upload-fingerprint")
async def upload_fingerprint(
    fingerprint: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    enable_preprocessing: bool = Form(True),
    generate_thumbnail: bool = Form(True)
):
    """Upload a fingerprint image to S3 with a unique file key"""
    try:
        if not fingerprint_matcher:
            raise HTTPException(status_code=500, detail="Fingerprint matcher not initialized")

        # Validate file type
        if fingerprint.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG, JPG, PNG, and TIFF are allowed."
            )

        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if fingerprint.size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {max_size / (1024*1024)}MB"
            )

        start_time = datetime.now()
        logger.info(f"Processing fingerprint upload - File size: {fingerprint.size} bytes")

        # Read file content
        file_content = await fingerprint.read()

        # Generate unique file key
        file_key = generate_unique_file_key(
            original_filename=fingerprint.filename,
            user_id=user_id,
            file_content=file_content
        )

        # Save file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            temp_path = temp_file.name

        try:
            # Load and validate image
            img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise HTTPException(status_code=400, detail="Failed to load image. Please check if the file is a valid image.")

            # Validate image dimensions
            height, width = img.shape
            if height < 100 or width < 100:
                raise HTTPException(
                    status_code=400, 
                    detail="Image too small. Minimum dimensions are 100x100 pixels."
                )
            if height > 4000 or width > 4000:
                raise HTTPException(
                    status_code=400, 
                    detail="Image too large. Maximum dimensions are 4000x4000 pixels."
                )

            # Preprocess image if enabled
            if enable_preprocessing:
                img = preprocess_fingerprint_image(img)
                logger.info("Image preprocessing completed")

            # Generate thumbnail if enabled
            thumbnail_key = None
            if generate_thumbnail:
                thumbnail_key = generate_thumbnail_key(file_key)
                thumbnail_img = generate_thumbnail_image(img)
                await upload_image_to_s3(thumbnail_img, thumbnail_key, "thumbnail")
                logger.info(f"Thumbnail uploaded: {thumbnail_key}")

            # Upload main image to S3
            await upload_image_to_s3(img, file_key, "main")
            logger.info(f"Main image uploaded: {file_key}")

            # Extract metadata
            metadata = extract_fingerprint_metadata(img, file_content)

            # Store metadata in S3 (as JSON)
            metadata_key = generate_metadata_key(file_key)
            metadata_content = {
                "upload_timestamp": datetime.now().isoformat(),
                "original_filename": fingerprint.filename,
                "file_size": fingerprint.size,
                "user_id": user_id,
                "description": description,
                "category": category,
                "image_dimensions": {
                    "width": width,
                    "height": height
                },
                "thumbnail_key": thumbnail_key,
                "metadata_key": metadata_key,
                "preprocessing_applied": enable_preprocessing,
                "thumbnail_generated": generate_thumbnail,
                **metadata
            }

            await upload_metadata_to_s3(metadata_content, metadata_key)
            logger.info(f"Metadata uploaded: {metadata_key}")

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return {
                "success": True,
                "upload": {
                    "file_key": file_key,
                    "thumbnail_key": thumbnail_key,
                    "metadata_key": metadata_key,
                    "original_filename": fingerprint.filename,
                    "file_size": fingerprint.size,
                    "upload_timestamp": metadata_content["upload_timestamp"],
                    "processing_time": processing_time
                },
                "image_info": {
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 3),
                    "file_size_mb": round(fingerprint.size / (1024 * 1024), 3)
                },
                "metadata": metadata_content
            }

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error in upload fingerprint endpoint: {error}")
        raise HTTPException(status_code=500, detail=str(error))

def generate_unique_file_key(original_filename: str, user_id: Optional[str], file_content: bytes) -> str:
    """Generate a unique file key for S3 storage"""
    # Generate hash from file content for consistency
    content_hash = hashlib.md5(file_content).hexdigest()[:8]
    
    # Generate UUID for uniqueness
    unique_id = str(uuid.uuid4())[:8]
    
    # Get file extension
    file_extension = Path(original_filename).suffix if original_filename else '.png'
    if not file_extension:
        file_extension = '.png'
    
    # Create timestamp component
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build the key
    if user_id:
        key = f"fingerprints/{user_id}/{timestamp}_{content_hash}_{unique_id}{file_extension}"
    else:
        key = f"fingerprints/{timestamp}_{content_hash}_{unique_id}{file_extension}"
    
    return key

def generate_thumbnail_key(file_key: str) -> str:
    """Generate thumbnail key from main file key"""
    path = Path(file_key)
    return str(path.parent / "thumbnails" / f"{path.stem}_thumb{path.suffix}")

def generate_metadata_key(file_key: str) -> str:
    """Generate metadata key from main file key"""
    path = Path(file_key)
    return str(path.parent / "metadata" / f"{path.stem}.json")

def preprocess_fingerprint_image(img: np.ndarray) -> np.ndarray:
    """Preprocess fingerprint image for better quality"""
    # Resize to standard size if too large
    max_size = 1024
    height, width = img.shape
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Apply bilateral filter to reduce noise while preserving edges
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Apply morphological operations to enhance ridge structure
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    return img

def generate_thumbnail_image(img: np.ndarray, size: tuple = (200, 200)) -> np.ndarray:
    """Generate a thumbnail version of the fingerprint image"""
    # Resize image to thumbnail size
    thumbnail = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    
    # Apply slight blur for thumbnail
    thumbnail = cv2.GaussianBlur(thumbnail, (3, 3), 0)
    
    return thumbnail

def extract_fingerprint_metadata(img: np.ndarray, file_content: bytes) -> Dict[str, Any]:
    """Extract metadata from fingerprint image"""
    height, width = img.shape
    
    # Calculate basic statistics
    mean_intensity = float(np.mean(img))
    std_intensity = float(np.std(img))
    min_intensity = float(np.min(img))
    max_intensity = float(np.max(img))
    
    # Calculate image quality metrics
    # Laplacian variance for sharpness
    laplacian_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
    
    # Calculate histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten().astype(float)
    
    # Calculate entropy
    hist_normalized = hist / hist.sum()
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
    
    # Calculate file hash
    file_hash = hashlib.sha256(file_content).hexdigest()
    
    return {
        "image_statistics": {
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "min_intensity": int(min_intensity),
            "max_intensity": int(max_intensity),
            "sharpness": round(laplacian_var, 2),
            "entropy": round(entropy, 2)
        },
        "file_hash": file_hash,
        "histogram": hist.tolist()
    }

async def upload_image_to_s3(img: np.ndarray, key: str, image_type: str) -> None:
    """Upload image to S3"""
    try:
        # Encode image to bytes
        success, encoded_img = cv2.imencode('.png', img)
        if not success:
            raise Exception("Failed to encode image")
        
        image_bytes = encoded_img.tobytes()
        
        # Upload to S3
        fingerprint_matcher.s3_client.put_object(
            Bucket=fingerprint_matcher.bucket_name,
            Key=key,
            Body=image_bytes,
            ContentType='image/png',
            Metadata={
                'image-type': image_type,
                'upload-timestamp': datetime.now().isoformat()
            }
        )
        
        logger.info(f"Successfully uploaded {image_type} image to S3: {key}")
        
    except Exception as e:
        logger.error(f"Failed to upload {image_type} image to S3: {e}")
        raise

async def upload_metadata_to_s3(metadata: Dict[str, Any], key: str) -> None:
    """Upload metadata to S3 as JSON"""
    try:
        import json
        
        # Convert metadata to JSON
        metadata_json = json.dumps(metadata, indent=2, default=str)
        
        # Upload to S3
        fingerprint_matcher.s3_client.put_object(
            Bucket=fingerprint_matcher.bucket_name,
            Key=key,
            Body=metadata_json.encode('utf-8'),
            ContentType='application/json',
            Metadata={
                'content-type': 'metadata',
                'upload-timestamp': datetime.now().isoformat()
            }
        )
        
        logger.info(f"Successfully uploaded metadata to S3: {key}")
        
    except Exception as e:
        logger.error(f"Failed to upload metadata to S3: {e}")
        raise

# S3-batched MCC endpoint
async def mcc_process_single_image(
    key: str,
    query_features: Dict[str, Any],
    threshold: float,
    min_inliers: int,
    num_sectors: int,
    num_rings: int,
    radius_px: int
) -> Optional[Dict[str, Any]]:
    try:
        try:
            response = fingerprint_matcher.s3_client.get_object(Bucket=fingerprint_matcher.bucket_name, Key=key)
            image_buffer = response['Body'].read()
        except Exception as e:
            logger.error(f"Failed to download image {key}: {e}")
            return None

        nparr = np.frombuffer(image_buffer, np.uint8)
        target_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if target_img is None:
            return None
        target_img = cv2.resize(target_img, (512, 512))

        s3_features = MCCMatcher.extract_features(target_img, num_sectors=num_sectors, num_rings=num_rings, radius=radius_px)
        comp = MCCMatcher.compare_features(query_features, s3_features, min_inliers_required=min_inliers)
        similarity = float(comp['similarity'])
        logger.info(f"mcc-minutiae: key={key} similarity={similarity:.3f} inliers={comp['inliers']} radius_px={radius_px}")

        if similarity >= threshold:
            return {
                "key": key,
                "filename": key,
                "similarity": similarity,
                "size": len(image_buffer),
                "match_quality": {
                    "excellent_match": similarity >= 0.90,
                    "very_good_match": similarity >= 0.80,
                    "good_match": similarity >= 0.70,
                    "fair_match": similarity >= 0.60,
                    "poor_match": similarity < 0.60
                },
                "analysis_details": {
                    "inliers": int(comp['inliers']),
                    "transform": comp['transform']
                }
            }

        return None

    except Exception as error:
        logger.error(f"Error processing image {key}: {error}")
        return None

@app.post("/mcc-minutiae-match")
async def mcc_minutiae_match(
    fingerprint: UploadFile = File(...),
    threshold: Optional[float] = Form(0.50),
    batch_size: Optional[int] = Form(50),
    max_results: Optional[int] = Form(10),
    min_inliers: Optional[int] = Form(14),
    dpi: Optional[int] = Form(512),
    radius_mm: Optional[float] = Form(5.0),
    num_rings: Optional[int] = Form(5),
    num_sectors: Optional[int] = Form(16)
):
    """MCC-inspired minutiae matching against S3 for highly accurate identification."""
    try:
        if not fingerprint_matcher:
            raise HTTPException(status_code=500, detail="Fingerprint matcher not initialized")

        valid_types = ["image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif"]
        if fingerprint.content_type not in valid_types:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, JPG, PNG, and TIFF are allowed.")

        start_time = datetime.now()
        file_content = await fingerprint.read()
        nparr = np.frombuffer(file_content, np.uint8)
        query_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if query_img is None:
            raise HTTPException(status_code=400, detail="Failed to load query image")
        query_img = cv2.resize(query_img, (512, 512))

        # Compute radius in pixels from DPI and desired mm
        # inches_per_mm = 1 / 25.4; pixels_per_inch = dpi
        # radius_mm * dpi / 25.4 -> pixels
        dpi_val = dpi or 512
        radius_mm_val = radius_mm or 5.0
        radius_px = int(max(20, min(160, round(radius_mm_val * dpi_val / 25.4))))
        rings_val = num_rings or 5
        sectors_val = num_sectors or 16

        logger.info(f"mcc-minutiae: dpi={dpi_val} radius_mm={radius_mm_val} -> radius_px={radius_px} rings={rings_val} sectors={sectors_val}")

        # Extract query MCC features with DPI-aware radius
        query_features = MCCMatcher.extract_features(query_img, num_sectors=sectors_val, num_rings=rings_val, radius=radius_px)

        all_keys = await fingerprint_matcher.get_all_fingerprint_keys()
        if all_keys is None or not all_keys:
            return {
                "success": True,
                "query": {
                    "file_size": fingerprint.size,
                    "file_name": fingerprint.filename
                },
                "results": {
                    "matches": [],
                    "total_processed": 0,
                    "processing_time": 0,
                    "total_time": 0
                },
                "configuration": {
                    "threshold": threshold,
                    "batch_size": batch_size or 50,
                    "min_inliers": min_inliers or 14
                }
            }

        if batch_size is None:
            batch_size = 50
        if threshold is None:
            threshold = 0.50
        if min_inliers is None:
            min_inliers = 14

        total_processed = 0
        matches: List[Dict[str, Any]] = []

        for i in range(0, len(all_keys), batch_size):
            batch_keys = all_keys[i:i + batch_size]
            tasks = [mcc_process_single_image(key, query_features, threshold, min_inliers, sectors_val, rings_val, radius_px) for key in batch_keys]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, dict) and result.get('similarity', 0) >= threshold:
                    matches.append(result)
                total_processed += 1

        matches.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        if max_results and len(matches) > max_results:
            matches = matches[:max_results]

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        response = {
            "success": True,
            "query": {
                "file_size": fingerprint.size,
                "file_name": fingerprint.filename
            },
            "results": {
                "matches": matches,
                "total_processed": total_processed,
                "processing_time": processing_time,
                "total_time": processing_time
            },
            "configuration": {
                "threshold": threshold,
                "batch_size": batch_size,
                "min_inliers": min_inliers,
                "dpi": dpi_val,
                "radius_mm": radius_mm_val,
                "radius_px": radius_px,
                "num_rings": rings_val,
                "num_sectors": sectors_val
            }
        }
        logger.info(f"Advanced minutiae match response: {response}")
        return response

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error in mcc-minutiae-match endpoint: {error}")
        raise HTTPException(status_code=500, detail=str(error))

async def ultra_minutiae_process_single_image(
    key: str,
    query_features: Dict[str, Any],
    min_inliers: int,
    threshold: float
) -> Optional[Dict[str, Any]]:
    """Process a single S3 image using the MinutiaeVerification algorithm."""
    try:
        try:
            response = fingerprint_matcher.s3_client.get_object(Bucket=fingerprint_matcher.bucket_name, Key=key)
            image_buffer = response['Body'].read()
        except Exception as e:
            logger.error(f"Failed to download image {key}: {e}")
            return None

        # Decode to grayscale image
        nparr = np.frombuffer(image_buffer, np.uint8)
        target_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if target_img is None:
            return None

        target_img = cv2.resize(target_img, (512, 512))
        s3_features = MinutiaeVerification.extract_features(target_img)

        comp = MinutiaeVerification.compare_features(query_features, s3_features, min_inliers_required=min_inliers)
        similarity = float(comp['score'])
        logger.info(f"ultra-minutiae: key={key} similarity={similarity:.3f} inliers={comp['inliers']}")

        if similarity >= threshold:
            return {
                "key": key,
                "filename": key,
                "similarity": similarity,
                "size": len(image_buffer),
                "match_quality": {
                    "excellent_match": bool(similarity >= 0.85),
                    "very_good_match": bool(similarity >= 0.75),
                    "good_match": bool(similarity >= 0.65),
                    "fair_match": bool(similarity >= 0.55),
                    "poor_match": bool(similarity < 0.55)
                },
                "analysis_details": {
                    "minutiae_count": int(s3_features['counts']['minutiae']),
                    "inliers": int(comp['inliers'])
                }
            }

        return None

    except Exception as error:
        logger.error(f"Error processing image {key}: {error}")
        return None

@app.post("/ultra-minutiae-match")
async def ultra_minutiae_match(
    fingerprint: UploadFile = File(...),
    threshold: Optional[float] = Form(0.65),
    batch_size: Optional[int] = Form(50),
    max_results: Optional[int] = Form(10),
    min_inliers: Optional[int] = Form(12)
):
    """Highly efficient minutiae-based matching against S3 using RANSAC alignment and inlier scoring."""
    try:
        if not fingerprint_matcher:
            raise HTTPException(status_code=500, detail="Fingerprint matcher not initialized")

        # Validate file type
        if fingerprint.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG, JPG, PNG, and TIFF are allowed."
            )

        start_time = datetime.now()
        logger.info(f"Processing ultra minutiae match request - File size: {fingerprint.size} bytes")

        # Read and decode query image
        file_content = await fingerprint.read()
        nparr = np.frombuffer(file_content, np.uint8)
        query_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if query_img is None:
            raise HTTPException(status_code=400, detail="Failed to load query image")
        query_img = cv2.resize(query_img, (512, 512))

        # Extract query features once
        query_features = MinutiaeVerification.extract_features(query_img)
        query_minutiae_count = query_features['counts']['minutiae']

        # Get all keys from S3
        all_keys = await fingerprint_matcher.get_all_fingerprint_keys()
        if all_keys is None or not all_keys:
            logger.warning("No fingerprint keys found in S3 bucket")
            return {
                "success": True,
                "query": {
                    "file_size": fingerprint.size,
                    "file_name": fingerprint.filename,
                    "minutiae_count": int(query_minutiae_count)
                },
                "results": {
                    "matches": [],
                    "total_processed": 0,
                    "processing_time": 0,
                    "total_time": 0
                },
                "configuration": {
                    "threshold": threshold,
                    "batch_size": batch_size or 50,
                    "min_inliers": min_inliers or 12
                }
            }

        # Defaults
        if batch_size is None:
            batch_size = 50
        if threshold is None:
            threshold = 0.65
        if min_inliers is None:
            min_inliers = 12

        total_processed = 0
        matches: List[Dict[str, Any]] = []

        # Process S3 keys in batches concurrently
        for i in range(0, len(all_keys), batch_size):
            batch_keys = all_keys[i:i + batch_size]
            tasks = [
                ultra_minutiae_process_single_image(key, query_features, min_inliers, threshold)
                for key in batch_keys
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, dict) and result.get('similarity', 0) >= threshold:
                    matches.append(result)
                total_processed += 1

        # Sort and trim
        matches.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        if max_results and len(matches) > max_results:
            matches = matches[:max_results]

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        response = {
            "success": True,
            "query": {
                "file_size": fingerprint.size,
                "file_name": fingerprint.filename,
                "minutiae_count": int(query_minutiae_count)
            },
            "results": {
                "matches": matches,
                "total_processed": total_processed,
                "processing_time": processing_time,
                "total_time": processing_time
            },
            "configuration": {
                "threshold": threshold,
                "batch_size": batch_size,
                "min_inliers": min_inliers
            }
        }
        logger.info(f"Ultra minutiae match response: {response}")
        return response

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error in ultra minutiae match endpoint: {error}")
        raise HTTPException(status_code=500, detail=str(error))

@app.post("/verify-minutiae")
async def verify_minutiae(
    fingerprint1: UploadFile = File(...),
    fingerprint2: UploadFile = File(...),
    threshold: Optional[float] = Form(0.65),
    min_inliers: Optional[int] = Form(12)
):
    """Verify whether two uploaded fingerprint images belong to the same finger using advanced minutiae matching."""
    try:
        valid_types = ["image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif"]
        if fingerprint1.content_type not in valid_types or fingerprint2.content_type not in valid_types:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, JPG, PNG, and TIFF are allowed.")

        start_time = datetime.now()
        logger.info(
            f"Processing minutiae verification - File1 size: {fingerprint1.size} bytes, File2 size: {fingerprint2.size} bytes"
        )

        file_content1 = await fingerprint1.read()
        file_content2 = await fingerprint2.read()

        def _decode_image(buf: bytes) -> np.ndarray:
            nparr = np.frombuffer(buf, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            return img

        img1 = _decode_image(file_content1)
        img2 = _decode_image(file_content2)
        if img1 is None or img2 is None:
            raise HTTPException(status_code=400, detail="Failed to load one or both images")

        img1 = cv2.resize(img1, (512, 512))
        img2 = cv2.resize(img2, (512, 512))

        verification = MinutiaeVerification.verify(img1, img2, min_inliers_required=min_inliers or 12)
        is_match = bool(verification['is_match'] and verification['score'] >= (threshold or 0.65))

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        response = {
            "success": True,
            "query": {
                "file1": {
                    "file_size": fingerprint1.size,
                    "file_name": fingerprint1.filename
                },
                "file2": {
                    "file_size": fingerprint2.size,
                    "file_name": fingerprint2.filename
                }
            },
            "results": {
                "is_match": is_match,
                "similarity": verification['score'],
                "inliers": verification['inliers'],
                "minutiae_counts": verification['minutiae_counts'],
                "transform": verification['transform'],
                "processing_time": total_time
            },
            "configuration": {
                "threshold": threshold,
                "min_inliers": min_inliers
            }
        }
        logger.info(f"Minutiae verification response: {response}")
        return response

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error in verify-minutiae endpoint: {error}")
        raise HTTPException(status_code=500, detail=str(error))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    ) 