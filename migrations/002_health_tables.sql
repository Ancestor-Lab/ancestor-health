-- Migration 002: Ancestor Health audit tables
-- Append-only. REVOKE UPDATE, DELETE from application role.

CREATE TABLE IF NOT EXISTS health_verification_sessions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    client_id           TEXT NOT NULL,
    input_text          TEXT NOT NULL,
    input_type          TEXT NOT NULL CHECK (input_type IN (
                            'ai_health_response', 'ai_clinical_summary',
                            'ai_patient_guidance', 'ai_policy_summary',
                            'lab_result_explanation'
                        )),
    input_context       TEXT,
    total_claims        INTEGER NOT NULL,
    overall_status      TEXT NOT NULL CHECK (overall_status IN (
                            'ALL_VERIFIED', 'PARTIALLY_VERIFIED',
                            'CONTRADICTION_DETECTED', 'VERIFICATION_FAILURE',
                            'ASYNC_PENDING'
                        )),
    overall_trust_score NUMERIC(5,4),
    total_latency_ms    INTEGER,
    score_version       TEXT NOT NULL,
    job_id              UUID,
    completed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_hvs_client_id  ON health_verification_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_hvs_created_at ON health_verification_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_hvs_status     ON health_verification_sessions(overall_status);


CREATE TABLE IF NOT EXISTS health_claim_results (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id              UUID NOT NULL REFERENCES health_verification_sessions(id),
    verification_result_id  UUID,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Extracted claim
    claim_text              TEXT NOT NULL,
    claim_type              TEXT NOT NULL,
    claim_span_start        INTEGER,
    claim_span_end          INTEGER,
    extraction_confidence   NUMERIC(5,4),

    -- Evidence retrieval
    evidence_source_type    TEXT,
    evidence_url            TEXT,
    evidence_title          TEXT,
    evidence_doi            TEXT,
    evidence_publication_date DATE,
    evidence_snippet        TEXT,
    retrieval_latency_ms    INTEGER,

    -- Claim-level result
    claim_status            TEXT NOT NULL CHECK (claim_status IN (
                                'VERIFIED', 'CONTRADICTION',
                                'VERIFICATION_FAILURE', 'NO_EVIDENCE_FOUND'
                            )),
    claim_trust_score       NUMERIC(5,4),
    failure_reason          TEXT
);

CREATE INDEX IF NOT EXISTS idx_hcr_session_id ON health_claim_results(session_id);
CREATE INDEX IF NOT EXISTS idx_hcr_status     ON health_claim_results(claim_status);

-- Immutability: REVOKE UPDATE, DELETE from application role
-- Run these manually after applying the migration:
-- REVOKE UPDATE, DELETE ON health_verification_sessions FROM app_role;
-- REVOKE UPDATE, DELETE ON health_claim_results FROM app_role;
