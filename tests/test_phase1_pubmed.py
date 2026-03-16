"""
Phase 1 Test: PubMed Connector + Health Scoring Config
Submits 5 known medical claims to PubMed and prints results.
Also validates health scoring tiers and recency decay.
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.pubmed_connector import PubMedConnector, PubMedError
from services.health_scoring_config import get_health_authority_score, get_health_recency_score
from services.heuristic_scorer import HeuristicScorer


# --- Test 1: PubMed Connector ---

TEST_CLAIMS = [
    "Elevated white blood cell counts may indicate bacterial infection",
    "Metformin is the first-line treatment for type 2 diabetes",
    "Hemoglobin levels below 13.5 g/dL in adult males suggest anemia",
    "High LDL cholesterol is a risk factor for cardiovascular disease",
    "Normal platelet count ranges from 150,000 to 400,000 per microliter of blood",
]


async def test_pubmed_connector():
    print("=" * 70)
    print("TEST 1: PubMed Connector — 5 Medical Claims")
    print("=" * 70)

    connector = PubMedConnector()
    passed = 0

    try:
        for i, claim in enumerate(TEST_CLAIMS, 1):
            print(f"\n--- Claim {i} ---")
            print(f"Claim: {claim}")
            try:
                result = await connector.search(claim)
                print(f"Query: {result.query}")
                print(f"Total found: {result.total_found}")
                if result.best_article:
                    art = result.best_article
                    print(f"Top result: {art.title}")
                    print(f"PMID: {art.pmid}")
                    print(f"Journal: {art.journal}")
                    print(f"Date: {art.publication_date}")
                    print(f"DOI: {art.doi}")
                    print(f"URL: {art.source_url}")
                    print(f"Snippet: {result.content_snippet[:200]}...")
                    print(f"Relevance: {result.relevance_score}")
                    passed += 1
                else:
                    print("NO RESULTS")
            except PubMedError as e:
                print(f"ERROR: {e}")

        print(f"\n{'=' * 70}")
        print(f"PubMed Connector: {passed}/{len(TEST_CLAIMS)} claims returned results")
        print(f"{'=' * 70}")
    finally:
        await connector.close()

    return passed


# --- Test 2: Health Scoring Config ---

def test_health_authority_tiers():
    print("\n" + "=" * 70)
    print("TEST 2: Health Authority Tiers")
    print("=" * 70)

    test_urls = [
        ("https://pubmed.ncbi.nlm.nih.gov/12345678/", "HEALTH_TIER_1", 1.0),
        ("https://www.who.int/news/item/hiv", "HEALTH_TIER_1", 1.0),
        ("https://www.cdc.gov/tb/causes", "HEALTH_TIER_1", 1.0),
        ("https://www.nejm.org/doi/full/10.1056/example", "HEALTH_TIER_1", 1.0),
        ("https://www.thelancet.com/article/S0140-6736", "HEALTH_TIER_1", 1.0),
        ("https://www.cochranelibrary.com/cdsr/doi/10.1002", "HEALTH_TIER_1", 1.0),
        ("https://www.uptodate.com/contents/example", "HEALTH_TIER_2", 0.8),
        ("https://www.medscape.com/viewarticle/example", "HEALTH_TIER_2", 0.8),
        ("https://www.nature.com/nm/articles/example", "HEALTH_TIER_2", 0.8),
        ("https://health.gov.au/topics/example", "HEALTH_TIER_2", 0.8),
        ("https://www.example-hospital.edu/research", "HEALTH_TIER_3", 0.6),
        ("https://www.randomsite.com/health-tips", "HEALTH_UNTIERED", 0.4),
    ]

    passed = 0
    for url, expected_tier, expected_score in test_urls:
        result = get_health_authority_score(url)
        status = "PASS" if result["tier"] == expected_tier and result["score"] == expected_score else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] {url[:50]:50s} → {result['tier']} ({result['score']})")

    print(f"\nAuthority tiers: {passed}/{len(test_urls)} passed")
    return passed == len(test_urls)


def test_health_recency_decay():
    print("\n" + "=" * 70)
    print("TEST 3: Health Recency Decay")
    print("=" * 70)

    from datetime import date

    ref_date = date(2026, 3, 16)
    test_cases = [
        ("2025-06-01", 1.0, "RECENT"),       # < 2 years
        ("2024-06-01", 1.0, "RECENT"),        # < 2 years from 2026-03-16
        ("2022-01-01", 0.7, "MODERATE"),      # 2-5 years
        ("2020-01-01", 0.4, "OUTDATED"),      # > 5 years
        ("2015-06-01", 0.4, "OUTDATED"),      # > 5 years
        (None, 0.3, "UNKNOWN"),               # No date
    ]

    passed = 0
    for pub_date, expected_score, expected_cat in test_cases:
        result = get_health_recency_score(pub_date, ref_date)
        status = "PASS" if result["score"] == expected_score else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] date={str(pub_date):12s} → score={result['score']} category={result['category']}")

    print(f"\nRecency decay: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_heuristic_scorer_health_profile():
    print("\n" + "=" * 70)
    print("TEST 4: HeuristicScorer with scoring_profile='health'")
    print("=" * 70)

    scorer = HeuristicScorer()

    # Test provenance with health profile
    result = scorer.calculate_provenance_score("https://pubmed.ncbi.nlm.nih.gov/12345/", scoring_profile="health")
    assert result["tier"] == "HEALTH_TIER_1", f"Expected HEALTH_TIER_1, got {result['tier']}"
    assert result["score"] == 1.0
    print(f"  [PASS] Health provenance: PubMed → {result['tier']} ({result['score']})")

    # Test provenance with general profile (unchanged)
    result_general = scorer.calculate_provenance_score("https://pubmed.ncbi.nlm.nih.gov/12345/", scoring_profile="general")
    assert result_general["tier"] in ("TIER_1_ALLOWLIST", "TIER_1_TLD", "UNKNOWN_NEUTRAL"), f"Got {result_general['tier']}"
    print(f"  [PASS] General provenance: PubMed → {result_general['tier']} ({result_general['score']})")

    # Test recency with health profile
    result_recency = scorer.calculate_recency_score("2020-01-01", scoring_profile="health")
    assert result_recency["score"] == 0.4, f"Expected 0.4, got {result_recency['score']}"
    print(f"  [PASS] Health recency: 2020-01-01 → {result_recency['score']} ({result_recency['category']})")

    # Test recency with general profile (unchanged behavior)
    result_recency_gen = scorer.calculate_recency_score("2020-01-01", scoring_profile="general")
    assert result_recency_gen["score"] == 50, f"Expected 50, got {result_recency_gen['score']}"
    print(f"  [PASS] General recency: 2020-01-01 → {result_recency_gen['score']}")

    print(f"\nHeuristic scorer integration: All passed")
    return True


async def main():
    print("\n" + "#" * 70)
    print("# PHASE 1 VALIDATION: PubMed Connector + Health Scoring Config")
    print("#" * 70)

    # Run offline tests first
    tier_ok = test_health_authority_tiers()
    recency_ok = test_health_recency_decay()
    scorer_ok = test_heuristic_scorer_health_profile()

    # Run PubMed test (requires network)
    pubmed_count = await test_pubmed_connector()

    print("\n" + "#" * 70)
    print("# PHASE 1 SUMMARY")
    print("#" * 70)
    print(f"  Health authority tiers: {'PASS' if tier_ok else 'FAIL'}")
    print(f"  Health recency decay:   {'PASS' if recency_ok else 'FAIL'}")
    print(f"  Scorer integration:     {'PASS' if scorer_ok else 'FAIL'}")
    print(f"  PubMed connector:       {pubmed_count}/{len(TEST_CLAIMS)} claims found evidence")

    if tier_ok and recency_ok and scorer_ok and pubmed_count >= 3:
        print("\n  *** PHASE 1: PASSED ***")
    else:
        print("\n  *** PHASE 1: NEEDS ATTENTION ***")


if __name__ == "__main__":
    asyncio.run(main())
