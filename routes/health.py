"""
Ancestor Health Verification Endpoint.
POST /v1/health/verify — orchestrates claim extraction → evidence retrieval
→ per-claim verification → response assembly.

GET /v1/health/verify/jobs/{job_id} — polls async job results.
"""

import asyncio
import time
import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks

from models.health_models import (
    HealthVerifyRequest,
    HealthVerifyResponse,
    HealthVerifyAsyncResponse,
    HealthJobStatusResponse,
    VerificationSummary,
    ClaimResult,
    ClaimVerification,
    SourceInfo,
    ScoreBreakdown,
    SemanticResult,
    EvidenceRetrievalInfo,
)
from services.health_claim_extractor import HealthClaimExtractor, ExtractedClaim
from services.medical_evidence_retriever import MedicalEvidenceRetriever, EvidenceCandidate
from services.health_scoring_config import get_health_authority_score, get_health_recency_score

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/health", tags=["Health Verification"])

# In-memory job store (production would use Redis/DB)
_jobs: Dict[str, Dict[str, Any]] = {}

# Shared service instances
_extractor = HealthClaimExtractor()
_retriever = MedicalEvidenceRetriever()

# Sync latency budget (ms) — if processing will likely exceed this, go async
SYNC_BUDGET_MS = 800

# Score version
SCORE_VERSION = "1.4.2"


@router.post("/verify", status_code=200)
async def verify_health(request: HealthVerifyRequest, background_tasks: BackgroundTasks):
    """
    Verify AI-generated health text.
    Extracts claims, retrieves evidence, runs verification pipeline.

    Returns 200 (sync) if fast enough, or 202 (async) for multi-claim verification.
    """
    start_time = time.monotonic()
    verification_id = uuid.uuid4()

    input_data = request.input
    options = request.options

    # Step 1: Extract claims
    try:
        extraction_result = await _extractor.extract(
            text=input_data.text,
            input_type=input_data.input_type,
            context=input_data.context,
            max_claims=options.max_claims,
        )
    except Exception as e:
        logger.error(f"Claim extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Claim extraction failed: {e}")

    claims = extraction_result.claims
    if not claims:
        raise HTTPException(status_code=422, detail="No verifiable claims extracted from input text")

    # Decide sync vs async
    estimated_ms = len(claims) * 2000  # ~2s per claim (PubMed + NLI)
    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    if options.allow_async and estimated_ms > SYNC_BUDGET_MS and len(claims) > 1:
        # Async path
        job_id = uuid.uuid4()
        _jobs[str(job_id)] = {
            "status": "ASYNC_PENDING",
            "verification_id": str(verification_id),
            "result": None,
            "error": None,
        }

        background_tasks.add_task(
            _run_verification_job,
            job_id=job_id,
            verification_id=verification_id,
            claims=claims,
            input_data=input_data,
            options=options,
            start_time=start_time,
        )

        response = HealthVerifyAsyncResponse(
            verification_id=verification_id,
            job_id=job_id,
            claims_extracted=len(claims),
            poll_url=f"/v1/health/verify/jobs/{job_id}",
            estimated_completion_ms=estimated_ms,
        )
        # Return 202
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=202, content=response.model_dump(mode="json"))

    # Sync path — process all claims now
    try:
        result = await _verify_all_claims(
            verification_id=verification_id,
            claims=claims,
            input_data=input_data,
            options=options,
            start_time=start_time,
        )
        return result
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {e}")


@router.get("/verify/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll async verification job status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "COMPLETE":
        return HealthJobStatusResponse(
            job_id=uuid.UUID(job_id),
            status="COMPLETE",
            result=job["result"],
        )
    elif job["status"] == "FAILED":
        return HealthJobStatusResponse(
            job_id=uuid.UUID(job_id),
            status="FAILED",
            error=job["error"],
        )
    else:
        return HealthJobStatusResponse(
            job_id=uuid.UUID(job_id),
            status="ASYNC_PENDING",
        )


async def _run_verification_job(
    job_id: uuid.UUID,
    verification_id: uuid.UUID,
    claims: list[ExtractedClaim],
    input_data,
    options,
    start_time: float,
):
    """Background task for async verification."""
    try:
        result = await _verify_all_claims(
            verification_id=verification_id,
            claims=claims,
            input_data=input_data,
            options=options,
            start_time=start_time,
        )
        _jobs[str(job_id)] = {
            "status": "COMPLETE",
            "verification_id": str(verification_id),
            "result": result,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Async verification job {job_id} failed: {e}")
        _jobs[str(job_id)] = {
            "status": "FAILED",
            "verification_id": str(verification_id),
            "result": None,
            "error": str(e),
        }


async def _verify_all_claims(
    verification_id: uuid.UUID,
    claims: list[ExtractedClaim],
    input_data,
    options,
    start_time: float,
) -> HealthVerifyResponse:
    """
    Run the full verification pipeline for all extracted claims.
    For each claim: retrieve evidence → NLI → trust score.
    """
    claim_results = []
    total_candidates = 0
    retrieval_latency = 0
    sources_queried_set = set()

    # Process claims sequentially to respect NCBI rate limits (10 req/s with API key).
    # Each claim does ESearch + EFetch (2-3 requests), so parallel would exceed limits.
    evidence_results = []
    for claim in claims:
        try:
            result = await _retriever.retrieve(
                claim=claim,
                sources=options.evidence_sources,
            )
            evidence_results.append(result)
        except Exception as e:
            evidence_results.append(e)
        await asyncio.sleep(0.2)  # Rate limit spacing

    # Process each claim's evidence through verification
    for claim, evidence_result in zip(claims, evidence_results):
        if isinstance(evidence_result, Exception):
            logger.warning(f"Evidence retrieval failed for claim '{claim.claim_text[:50]}': {evidence_result}")
            claim_results.append(_build_failure_result(claim, "EVIDENCE_RETRIEVAL_ERROR"))
            continue

        total_candidates += evidence_result.total_results
        retrieval_latency = max(retrieval_latency, evidence_result.retrieval_latency_ms)
        sources_queried_set.update(evidence_result.sources_queried)

        if not evidence_result.best_candidate:
            claim_results.append(_build_failure_result(claim, "NO_MATCHING_EVIDENCE"))
            continue

        # Run NLI + scoring on best evidence
        claim_result = await _verify_single_claim(claim, evidence_result.best_candidate, options)
        claim_results.append(claim_result)

    # Compute summary
    summary = _compute_summary(claim_results, options.min_trust_threshold)

    total_latency_ms = int((time.monotonic() - start_time) * 1000)

    return HealthVerifyResponse(
        verification_id=verification_id,
        status="COMPLETE",
        processing_mode="sync",
        input_type=input_data.input_type,
        summary=summary,
        claims=claim_results,
        evidence_retrieval=EvidenceRetrievalInfo(
            sources_queried=list(sources_queried_set),
            total_candidates_found=total_candidates,
            retrieval_latency_ms=retrieval_latency,
        ),
        ancestry_log_url=f"/v1/health/ancestry/{verification_id}",
        total_latency_ms=total_latency_ms,
        score_version=SCORE_VERSION,
    )


async def _verify_single_claim(
    claim: ExtractedClaim,
    evidence: EvidenceCandidate,
    options,
) -> ClaimResult:
    """
    Verify a single claim against its best evidence candidate.
    Runs NLI inference and computes multiplicative trust score.
    """
    try:
        # Get NLI service
        from services.nli_service import NLIService
        nli_svc = NLIService()

        # Run NLI: claim vs evidence snippet
        nli_result = nli_svc.analyze_claim_evidence(
            claim.claim_text,
            evidence.content_snippet,
        )

        probs = nli_result["raw_probabilities"]
        contradiction_prob = probs.get("contradiction", 0.0)
        entailment_prob = probs.get("entailment", 0.0)

        # Kill switch: contradiction > 0.5 → trust score = 0
        if contradiction_prob > 0.5:
            return ClaimResult(
                claim_id=uuid.UUID(claim.claim_id),
                claim_text=claim.claim_text,
                claim_type=claim.claim_type,
                extraction_confidence=claim.confidence,
                verification=ClaimVerification(
                    status="CONTRADICTION",
                    trust_score=0.0,
                    score_breakdown=ScoreBreakdown(
                        authority=0.0, availability=0.0, entailment=0.0,
                        formula="KILL SWITCH: contradiction > 0.5 → score = 0",
                    ),
                    source=_build_source_info(evidence),
                    semantic_result=SemanticResult(
                        label="CONTRADICTION",
                        confidence=contradiction_prob,
                    ),
                ),
            )

        # Compute multiplicative trust score: Authority × Availability × Entailment
        authority_result = get_health_authority_score(evidence.source_url)
        authority = authority_result["score"]

        availability = 1.0  # Evidence was successfully retrieved

        entailment = entailment_prob  # Raw NLI entailment probability

        trust_score = authority * availability * entailment

        # Determine status
        if trust_score >= options.min_trust_threshold:
            status = "VERIFIED"
        elif entailment_prob <= 0.4:
            status = "VERIFICATION_FAILURE"
        else:
            status = "VERIFIED"  # Above threshold counts as verified

        return ClaimResult(
            claim_id=uuid.UUID(claim.claim_id),
            claim_text=claim.claim_text,
            claim_type=claim.claim_type,
            extraction_confidence=claim.confidence,
            verification=ClaimVerification(
                status=status,
                trust_score=round(trust_score, 4),
                score_breakdown=ScoreBreakdown(
                    authority=round(authority, 4),
                    availability=availability,
                    entailment=round(entailment, 4),
                ),
                source=_build_source_info(evidence),
                semantic_result=SemanticResult(
                    label=nli_result["predicted_label"],
                    confidence=round(nli_result["confidence_score"], 4),
                ),
            ),
        )

    except Exception as e:
        logger.error(f"Single claim verification failed: {e}")
        return _build_failure_result(claim, f"NLI_ERROR: {e}")


def _build_source_info(evidence: EvidenceCandidate) -> SourceInfo:
    """Build SourceInfo from an EvidenceCandidate."""
    authority_result = get_health_authority_score(evidence.source_url)
    return SourceInfo(
        url=evidence.source_url,
        title=evidence.title,
        source_type=evidence.source_type,
        domain_tier=authority_result["tier"],
        publication_date=evidence.publication_date,
        doi=evidence.doi,
    )


def _build_failure_result(claim: ExtractedClaim, reason: str) -> ClaimResult:
    """Build a VERIFICATION_FAILURE result for a claim."""
    # Determine if it's a no-evidence or other failure
    status = "NO_EVIDENCE_FOUND" if "NO_MATCHING" in reason else "VERIFICATION_FAILURE"

    return ClaimResult(
        claim_id=uuid.UUID(claim.claim_id),
        claim_text=claim.claim_text,
        claim_type=claim.claim_type,
        extraction_confidence=claim.confidence,
        verification=ClaimVerification(
            status=status,
            trust_score=None,
            failure_reason=reason,
        ),
    )


def _compute_summary(
    claim_results: list[ClaimResult],
    min_trust_threshold: float,
) -> VerificationSummary:
    """Compute overall verification summary from claim results."""
    total = len(claim_results)
    verified = 0
    contradicted = 0
    unverifiable = 0
    scores = []

    for cr in claim_results:
        v = cr.verification
        if v.status == "VERIFIED":
            verified += 1
            if v.trust_score is not None:
                scores.append(v.trust_score)
        elif v.status == "CONTRADICTION":
            contradicted += 1
        else:
            unverifiable += 1

    # Overall status logic (Section 4.4)
    if contradicted > 0:
        overall_status = "CONTRADICTION_DETECTED"
    elif verified == total:
        overall_status = "ALL_VERIFIED"
    elif verified > 0:
        overall_status = "PARTIALLY_VERIFIED"
    else:
        overall_status = "VERIFICATION_FAILURE"

    overall_trust = round(sum(scores) / len(scores), 4) if scores else None

    return VerificationSummary(
        total_claims=total,
        verified=verified,
        contradicted=contradicted,
        unverifiable=unverifiable,
        overall_trust_score=overall_trust,
        overall_status=overall_status,
    )
