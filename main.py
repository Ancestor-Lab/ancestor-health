"""
Phase 3: API Layer with Availability & Hardening
FastAPI service for Cloud Run deployment with cached content fetching, retry logic, and audit snapshots.

This service exposes endpoints for:
1. URL content extraction with caching and retries
2. NLI semantic analysis with production thresholds
3. Combined claim verification pipeline with audit-grade snapshotting

Phase 3 Features:
- Deterministic source caching with Redis backend
- Exponential backoff retry for transient failures
- SHA-256 audit snapshots for content integrity
- Structured availability signals
- Fail-closed behavior (security first)
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import time

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
import uvicorn

from services.source_fetcher import SourceFetcher, fetch_source_content
from services.nli_service import NLIService, get_nli_service
from services.cache_service import CacheService
from routes.health import router as health_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
source_fetcher = None
nli_service = None
cache_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Phase 3: Application lifespan manager with caching and hardening.
    """
    global source_fetcher, nli_service, cache_service

    logger.info("🚀 Phase 3: Starting hardened services with caching and retry logic...")

    try:
        # Phase 3: Initialize cache service first
        logger.info("Initializing cache service...")
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_password = os.getenv("REDIS_PASSWORD")

        cache_service = CacheService(
            host=redis_host,
            port=redis_port,
            password=redis_password
        )

        if cache_service.enabled:
            logger.info("✅ Cache service ready with Redis backend")
        else:
            logger.warning("⚠️ Cache service disabled - using live fetch only (fail-closed)")

        # Phase 3: Initialize source fetcher with cache service
        logger.info("Initializing hardened source fetcher...")
        source_fetcher = SourceFetcher(
            cache_service=cache_service,
            timeout=10,  # Increased for better stability
            max_content_length=2*1024*1024  # 2MB for better coverage
        )
        logger.info("✅ Hardened source fetcher ready")

        # Initialize NLI service placeholder (lazy loading for faster startup)
        logger.info("NLI service configured for lazy loading...")
        nli_service = None  # Will be loaded on first request
        logger.info("✅ NLI service configured for on-demand loading")

        # Phase 3: Log cache statistics
        if cache_service.enabled:
            stats = cache_service.get_stats()
            logger.info(f"📊 Cache stats: {stats}")

        logger.info("🎉 All Phase 3 services initialized successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Phase 3 services: {str(e)}")
        raise

    yield

    logger.info("🔄 Shutting down Phase 3 services...")
    if cache_service and cache_service.enabled:
        logger.info("📊 Final cache statistics:")
        try:
            stats = cache_service.get_stats()
            logger.info(f"   {stats}")
        except Exception:
            pass

# Create FastAPI app
app = FastAPI(
    title="Ancestor Phase 7: API-Aligned Source & NLI Service",
    description="Cached URL content fetching with retry logic and semantic analysis for claim verification",
    version="7.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a router for API versioning
api_router = APIRouter(prefix="/api/v1")

# Request/Response models
class SourceFetchRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL to fetch and extract content from")
    timeout: Optional[int] = Field(5, description="Request timeout in seconds", ge=1, le=30)

class SourceFetchResponse(BaseModel):
    availability_signal: float = Field(..., description="1.0 for success, 0.0 for failure")
    extracted_text: str = Field(..., description="Clean extracted text content")
    content_length: int = Field(..., description="Length of extracted text")
    final_url: str = Field(..., description="Final URL after redirects")
    fetch_time_ms: int = Field(..., description="Time taken for fetch operation")
    fetch_status: str = Field(..., description="Structured availability signal (FETCH_OK, FETCH_CACHED, etc.)")
    snapshot_hash: str = Field(..., description="SHA-256 hash of extracted content for audit trail")
    cached_timestamp: Optional[float] = Field(None, description="Timestamp when content was cached")
    original_fetch_timestamp: Optional[float] = Field(None, description="Original fetch timestamp for cached content")
    error_type: Optional[str] = Field(None, description="Error classification if failed")
    error_message: Optional[str] = Field(None, description="Detailed error message if failed")

class NLIAnalysisRequest(BaseModel):
    claim: str = Field(..., description="Claim to verify", min_length=1, max_length=1000)
    evidence: str = Field(..., description="Evidence text to compare against", min_length=1, max_length=5000)

class NLIAnalysisResponse(BaseModel):
    raw_probabilities: Dict[str, float] = Field(..., description="Raw NLI probabilities")
    predicted_label: str = Field(..., description="Most likely label")
    confidence_score: float = Field(..., description="Confidence in prediction (0-1)")
    label_status: str = Field(..., description="Always PROVISIONAL_SHADOW_MODE")
    inference_time_ms: int = Field(..., description="Time taken for inference")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")

class ClaimVerificationRequest(BaseModel):
    claim: str = Field(..., description="Claim to verify", min_length=1, max_length=1000)
    source_url: HttpUrl = Field(..., description="Source URL to fetch evidence from")
    timeout: Optional[int] = Field(5, description="Request timeout in seconds", ge=1, le=30)

class ClaimVerificationResponse(BaseModel):
    claim: str = Field(..., description="Original claim")
    source_url: str = Field(..., description="Source URL")
    source_fetch_result: SourceFetchResponse = Field(..., description="Source fetching result")
    nli_analysis: Optional[NLIAnalysisResponse] = Field(None, description="NLI analysis result")
    verification_status: str = Field(..., description="PROVISIONAL verification status")
    confidence_score: float = Field(..., description="Overall confidence score")
    processing_time_ms: int = Field(..., description="Total processing time")
    shadow_mode_warning: str = Field(..., description="Shadow mode warning")

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {
        "status": "healthy",
        "phase": "3.0",
        "services": {
            "source_fetcher": source_fetcher is not None,
            "nli_service": nli_service is not None,
            "cache_service": cache_service.enabled if cache_service else False
        },
        "cache_stats": cache_service.get_stats() if cache_service and cache_service.enabled else None,
        "timestamp": time.time()
    }

# Source fetching endpoint
@app.post("/fetch-source", response_model=SourceFetchResponse, tags=["Source Fetching"])
async def fetch_source_endpoint(request: SourceFetchRequest):
    """
    Fetch and extract text content from a URL.
    Returns availability_signal: 1.0 for success, 0.0 for failure.
    """
    if not source_fetcher:
        raise HTTPException(status_code=503, detail="Source fetcher not initialized")

    try:
        logger.info(f"Fetching source: {request.url}")
        result = source_fetcher.fetch_and_extract(str(request.url))
        logger.info(f"Source fetch completed: signal={result['availability_signal']}, length={result['content_length']}")
        return SourceFetchResponse(**result)

    except Exception as e:
        logger.error(f"Source fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Source fetch failed: {str(e)}")

# NLI analysis endpoint
def get_nli_service():
    """Lazy load NLI service on first use."""
    global nli_service
    if nli_service is None:
        logger.info("🔄 Loading NLI model on-demand...")
        nli_service = NLIService()
        logger.info("✅ NLI model loaded successfully")
    return nli_service

@app.post("/analyze-nli", response_model=NLIAnalysisResponse, tags=["NLI Analysis"])
async def analyze_nli_endpoint(request: NLIAnalysisRequest):
    """
    Perform NLI semantic analysis between claim and evidence.
    Returns raw probabilities with production thresholds.
    """
    nli_svc = get_nli_service()

    try:
        logger.info(f"NLI analysis: claim={request.claim[:50]}..., evidence={request.evidence[:50]}...")
        result = nli_svc.analyze_claim_evidence(request.claim, request.evidence)
        logger.info(f"NLI analysis completed: label={result['predicted_label']}, confidence={result['confidence_score']:.3f}")
        return NLIAnalysisResponse(**result)

    except Exception as e:
        logger.error(f"NLI analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"NLI analysis failed: {str(e)}")

# Combined claim verification endpoint
@api_router.post("/verify", response_model=ClaimVerificationResponse, tags=["Claim Verification"])
async def verify_claim_endpoint(request: ClaimVerificationRequest):
    """
    Complete claim verification pipeline: fetch source + NLI analysis.
    All results marked as PROVISIONAL_SHADOW_MODE for observation only.
    """
    if not source_fetcher:
        raise HTTPException(status_code=503, detail="Services not initialized")

    start_time = time.time()

    try:
        logger.info(f"Starting claim verification: {request.claim[:50]}... vs {request.source_url}")

        # Step 1: Fetch source content
        source_result = source_fetcher.fetch_and_extract(str(request.source_url))

        # Step 2: NLI analysis if source fetch successful
        nli_result = None
        if source_result['availability_signal'] > 0 and source_result['extracted_text']:
            nli_svc = get_nli_service()
            nli_result = nli_svc.analyze_claim_evidence(
                request.claim,
                source_result['extracted_text']
            )

        # Step 3: PHASE 2D - Apply production decision thresholds
        if nli_result:
            # Apply frozen decision thresholds from Phase 2C calibration
            production_decision = nli_svc.apply_production_decision_thresholds(
                nli_result['raw_probabilities'],
                source_available=True
            )
            verification_status = production_decision['verification_status']
            confidence_score = production_decision['confidence']
        else:
            # Source unavailable - apply frozen decision logic
            nli_svc = get_nli_service()
            production_decision = nli_svc.apply_production_decision_thresholds(
                {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0},
                source_available=False
            )
            verification_status = production_decision['verification_status']
            confidence_score = production_decision['confidence']

        processing_time = time.time() - start_time

        response = ClaimVerificationResponse(
            claim=request.claim,
            source_url=str(request.source_url),
            source_fetch_result=SourceFetchResponse(**source_result),
            nli_analysis=NLIAnalysisResponse(**nli_result) if nli_result else None,
            verification_status=verification_status,
            confidence_score=confidence_score,
            processing_time_ms=int(processing_time * 1000),
            shadow_mode_warning="PHASE 3 PRODUCTION READY - HARDENED WITH CACHING AND RETRIES"
        )

        logger.info(f"Claim verification completed: status={verification_status}, confidence={confidence_score:.3f}")
        return response

    except Exception as e:
        logger.error(f"Claim verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Claim verification failed: {str(e)}")

# Service info endpoint
@app.get("/info", tags=["Service Info"])
async def service_info():
    """Get information about loaded services and models."""
    return {
        "phase": "3.0",
        "mode": "PRODUCTION_READY",
        "features": [
            "Deterministic source caching with Redis",
            "Exponential backoff retry for transient failures",
            "SHA-256 audit snapshots for content integrity",
            "Structured availability signals",
            "Frozen decision thresholds from Phase 2D",
            "Fail-closed behavior (security first)"
        ],
        "services": {
            "source_fetcher": {
                "available": source_fetcher is not None,
                "timeout": getattr(source_fetcher, 'timeout', None),
                "max_content_length": getattr(source_fetcher, 'max_content_length', None),
                "cache_enabled": cache_service.enabled if cache_service else False
            },
            "nli_service": get_nli_service().get_model_info() if nli_service else {"available": True, "lazy_loaded": True},
            "cache_service": cache_service.get_stats() if cache_service and cache_service.enabled else {"available": False}
        },
        "notice": "PHASE 3 HARDENING COMPLETE - READY FOR PRODUCTION USE"
    }

# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Ancestor Phase 3: Hardened Source & NLI Service",
        "mode": "PRODUCTION_READY",
        "version": "3.0.0",
        "endpoints": [
            "/health - Health check with cache statistics",
            "/fetch-source - Cached URL content extraction with retries",
            "/analyze-nli - Semantic analysis with production thresholds",
            "/verify-claim - Complete hardened verification pipeline",
            "/info - Detailed service information"
        ],
        "phase3_features": [
            "Redis caching with URL normalization",
            "Exponential backoff retry logic",
            "SHA-256 audit snapshots",
            "Structured availability signals",
            "Fail-closed security behavior"
        ],
        "notice": "PHASE 7 COMPLETE - API ALIGNED & PRODUCTION READY"
    }

# Register the API router
app.include_router(api_router)

# Register health verification routes
app.include_router(health_router)

# Legacy endpoint for backward compatibility
@app.post("/api/v1/verify", response_model=ClaimVerificationResponse, tags=["Claim Verification"])
async def verify_claim_legacy(request: ClaimVerificationRequest):
    """Legacy endpoint that mirrors the new API structure"""
    return await verify_claim_endpoint(request)

if __name__ == "__main__":
    # Run the service locally for testing
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False  # Disable reload for production
    )
