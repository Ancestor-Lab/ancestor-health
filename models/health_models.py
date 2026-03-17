"""
Pydantic models for the Ancestor Health verification endpoint.
Request/response schemas per Architecture Doc Sections 4.1–4.4.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# --- Request Models ---

class HealthInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192, description="AI-generated health output to verify")
    input_type: str = Field(
        ...,
        description="Type of AI-generated input",
    )
    context: Optional[str] = Field(None, description="Optional clinical context")

    @field_validator("input_type")
    @classmethod
    def validate_input_type(cls, v: str) -> str:
        valid = {
            "ai_health_response",
            "ai_clinical_summary",
            "ai_patient_guidance",
            "ai_policy_summary",
            "lab_result_explanation",
        }
        if v not in valid:
            raise ValueError(f"input_type must be one of: {valid}")
        return v


class HealthOptions(BaseModel):
    callback_url: Optional[str] = None
    max_claims: int = Field(20, ge=1, le=50)
    evidence_sources: list[str] = Field(default_factory=lambda: ["pubmed", "who"])
    min_trust_threshold: float = Field(0.5, ge=0.0, le=1.0)
    allow_async: bool = True


class HealthVerifyRequest(BaseModel):
    input: HealthInput
    options: Optional[HealthOptions] = Field(default_factory=HealthOptions)


# --- Response Models ---

class SourceInfo(BaseModel):
    url: str
    title: str
    source_type: str
    domain_tier: Optional[str] = None
    publication_date: Optional[str] = None
    doi: Optional[str] = None


class ScoreBreakdown(BaseModel):
    authority: float
    availability: float
    entailment: float
    formula: str = "authority × availability × entailment"


class SemanticResult(BaseModel):
    label: str
    confidence: float
    nli_model_version: str = "cross-encoder/nli-distilroberta-base"


class ClaimVerification(BaseModel):
    verification_id: UUID = Field(default_factory=uuid4)
    status: str  # VERIFIED | PARTIAL_SUPPORT | INCONCLUSIVE | CONTRADICTION | VERIFICATION_FAILURE | NO_EVIDENCE_FOUND
    trust_score: Optional[float] = None
    score_breakdown: Optional[ScoreBreakdown] = None
    source: Optional[SourceInfo] = None
    semantic_result: Optional[SemanticResult] = None
    failure_reason: Optional[str] = None


class ClaimResult(BaseModel):
    claim_id: UUID
    claim_text: str
    claim_type: str
    extraction_confidence: float
    verification: ClaimVerification


class VerificationSummary(BaseModel):
    total_claims: int
    verified: int
    contradicted: int
    unverifiable: int
    overall_trust_score: Optional[float] = None
    overall_status: str  # ALL_VERIFIED | PARTIALLY_VERIFIED | CONTRADICTION_DETECTED | VERIFICATION_FAILURE


class EvidenceRetrievalInfo(BaseModel):
    sources_queried: list[str]
    total_candidates_found: int
    retrieval_latency_ms: int


class HealthVerifyResponse(BaseModel):
    """200 OK synchronous response."""
    verification_id: UUID = Field(default_factory=uuid4)
    status: str = "COMPLETE"
    processing_mode: str = "sync"
    input_type: str
    summary: VerificationSummary
    claims: list[ClaimResult]
    evidence_retrieval: EvidenceRetrievalInfo
    ancestry_log_url: Optional[str] = None
    total_latency_ms: int
    verified_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    score_version: str = "1.4.2"


class HealthVerifyAsyncResponse(BaseModel):
    """202 Accepted async response."""
    verification_id: UUID = Field(default_factory=uuid4)
    job_id: UUID = Field(default_factory=uuid4)
    status: str = "ASYNC_PENDING"
    processing_mode: str = "async"
    claims_extracted: int
    poll_url: str
    estimated_completion_ms: int
    queued_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HealthJobStatusResponse(BaseModel):
    """GET /v1/health/verify/jobs/{job_id} response."""
    job_id: UUID
    status: str  # ASYNC_PENDING | COMPLETE | FAILED
    result: Optional[HealthVerifyResponse] = None
    error: Optional[str] = None
