"""
Phase 2 Test: Claim Extractor + Evidence Retriever
Takes a sample AI-generated health explanation, extracts claims,
then retrieves PubMed evidence for each claim.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.health_claim_extractor import HealthClaimExtractor
from services.lab_explanation_handler import LabExplanationHandler
from services.medical_evidence_retriever import MedicalEvidenceRetriever
from services.pubmed_connector import PubMedError

# Sample AI-generated lab result explanation
SAMPLE_TEXT = """Your complete blood count shows a white blood cell count of 15.2 × 10³/µL, \
which is above the normal range of 4.5–11.0 × 10³/µL. This elevation may indicate an active \
bacterial infection or inflammatory response. Your hemoglobin level of 10.8 g/dL is slightly \
below the normal range for adult males (13.5–17.5 g/dL), which could suggest mild anemia. \
The platelet count of 245 × 10³/µL falls within the normal range of 150–400 × 10³/µL."""


async def test_lab_handler():
    print("=" * 70)
    print("TEST 1: Lab Explanation Handler — Biomarker Detection")
    print("=" * 70)

    handler = LabExplanationHandler()
    annotated = handler.preprocess(SAMPLE_TEXT)

    print(f"Found {len(annotated.biomarkers)} biomarker references:\n")
    for ref in annotated.biomarkers:
        print(f"  Biomarker: {ref.biomarker_name}")
        print(f"    Value: {ref.stated_value} {ref.stated_unit or ''}")
        print(f"    Range: {ref.stated_range}")
        print(f"    Span: {ref.span}")
        print()

    passed = len(annotated.biomarkers) >= 3
    print(f"Lab handler: {'PASS' if passed else 'FAIL'} ({len(annotated.biomarkers)} biomarkers found)")
    return passed


async def test_claim_extraction():
    print("\n" + "=" * 70)
    print("TEST 2: Health Claim Extractor")
    print("=" * 70)

    extractor = HealthClaimExtractor()
    result = await extractor.extract(
        text=SAMPLE_TEXT,
        input_type="lab_result_explanation",
        max_claims=10,
    )

    print(f"Extracted {len(result.claims)} claims in {result.extraction_latency_ms}ms:\n")
    for i, claim in enumerate(result.claims, 1):
        print(f"  Claim {i}:")
        print(f"    Text: {claim.claim_text}")
        print(f"    Type: {claim.claim_type}")
        print(f"    Confidence: {claim.confidence}")
        print(f"    Span: {claim.original_span}")
        print()

    passed = len(result.claims) >= 2
    print(f"Claim extraction: {'PASS' if passed else 'FAIL'} ({len(result.claims)} claims, {result.extraction_latency_ms}ms)")

    perf_ok = result.extraction_latency_ms < 200
    print(f"Performance (<200ms): {'PASS' if perf_ok else 'WARN'} ({result.extraction_latency_ms}ms)")

    return result, passed


async def test_evidence_retrieval(claims):
    print("\n" + "=" * 70)
    print("TEST 3: Evidence Retrieval (PubMed)")
    print("=" * 70)

    retriever = MedicalEvidenceRetriever()
    evidence_found = 0

    try:
        for i, claim in enumerate(claims, 1):
            print(f"\n--- Claim {i}: {claim.claim_text[:80]}... ---")
            try:
                result = await retriever.retrieve(
                    claim=claim,
                    sources=["pubmed"],  # PubMed only for this test
                )
                print(f"  Sources queried: {result.sources_queried}")
                print(f"  Total candidates: {result.total_results}")
                print(f"  Latency: {result.retrieval_latency_ms}ms")

                if result.best_candidate:
                    bc = result.best_candidate
                    print(f"  Best match:")
                    print(f"    Title: {bc.title}")
                    print(f"    URL: {bc.source_url}")
                    print(f"    Type: {bc.source_type}")
                    print(f"    Date: {bc.publication_date}")
                    print(f"    Relevance: {bc.relevance_score}")
                    print(f"    Snippet: {bc.content_snippet[:150]}...")
                    evidence_found += 1
                else:
                    print(f"  No evidence found")

            except PubMedError as e:
                print(f"  PubMed error: {e}")
    finally:
        await retriever.close()

    print(f"\nEvidence retrieval: {evidence_found}/{len(claims)} claims have evidence")
    return evidence_found


async def main():
    print("\n" + "#" * 70)
    print("# PHASE 2 VALIDATION: Claim Extraction + Evidence Retrieval")
    print("#" * 70)
    print(f"\nSample text ({len(SAMPLE_TEXT)} chars):")
    print(f"  {SAMPLE_TEXT[:200]}...\n")

    # Test 1: Lab handler
    lab_ok = await test_lab_handler()

    # Test 2: Claim extraction
    extraction_result, extract_ok = await test_claim_extraction()

    # Test 3: Evidence retrieval (network)
    evidence_count = 0
    if extraction_result.claims:
        evidence_count = await test_evidence_retrieval(extraction_result.claims)

    print("\n" + "#" * 70)
    print("# PHASE 2 SUMMARY")
    print("#" * 70)
    print(f"  Lab handler:         {'PASS' if lab_ok else 'FAIL'}")
    print(f"  Claim extraction:    {'PASS' if extract_ok else 'FAIL'} ({len(extraction_result.claims)} claims)")
    print(f"  Evidence retrieval:  {evidence_count}/{len(extraction_result.claims)} claims with evidence")

    if lab_ok and extract_ok and evidence_count >= 1:
        print("\n  *** PHASE 2: PASSED ***")
    else:
        print("\n  *** PHASE 2: NEEDS ATTENTION ***")


if __name__ == "__main__":
    asyncio.run(main())
