"""
Health-Specific Scoring Configuration for Ancestor Health.
Defines medical domain authority tiers and health recency decay.

Used by heuristic_scorer.py when scoring_profile="health".
See Architecture Doc Section 4.5 for exact values.
"""

from datetime import date
from typing import Dict, Any, Optional
from urllib.parse import urlparse

# Health Tier 1 (authority = 1.0): Top medical evidence sources
HEALTH_TIER_1_DOMAINS = {
    # WHO
    "who.int", "apps.who.int",
    # PubMed / PMC
    "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    # Cochrane
    "cochranelibrary.com", "cochrane.org",
    # NICE
    "nice.org.uk",
    # CDC
    "cdc.gov",
    # Top journals
    "thelancet.com", "nejm.org", "bmj.com", "jamanetwork.com",
}

# Health Tier 2 (authority = 0.8): Trusted clinical sources
HEALTH_TIER_2_DOMAINS = {
    # Clinical decision tools
    "uptodate.com", "medscape.com", "dynamed.com",
    # High-impact journals
    "nature.com", "cell.com",
}

# Health Tier 2 also includes national ministry of health .gov domains
# (detected by TLD + health context heuristic)

# Health Tier 3 (authority = 0.6): University hospitals, regional health authorities
HEALTH_TIER_3_PATTERNS = [
    "hospital", "health.gov", "healthauthority",
    "medicalcenter", "medical-center",
    ".ac.uk", ".edu",
]

# Medical association domains (Tier 3)
HEALTH_TIER_3_DOMAINS = {
    "ama-assn.org", "who.int",  # already in T1 but for completeness
    "aap.org", "acc.org", "acog.org", "asco.org",
}

# Authority scores by tier
HEALTH_AUTHORITY_SCORES = {
    "HEALTH_TIER_1": 1.0,
    "HEALTH_TIER_2": 0.8,
    "HEALTH_TIER_3": 0.6,
    "HEALTH_UNTIERED": 0.4,
}

# Health recency decay (stricter than general)
HEALTH_RECENCY = {
    "recent": {"max_years": 2, "score": 1.0},
    "moderate": {"max_years": 5, "score": 0.7},
    "outdated": {"max_years": float("inf"), "score": 0.4},
    "unknown": {"score": 0.3},
}


def get_health_authority_score(source_url: str) -> Dict[str, Any]:
    """
    Calculate health-specific authority score for a source URL.

    Returns:
        Dict with 'score' (0.0-1.0), 'tier', 'domain', 'reasoning'
    """
    try:
        parsed = urlparse(source_url.lower())
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]

        # Check Tier 1
        for t1_domain in HEALTH_TIER_1_DOMAINS:
            if domain == t1_domain or domain.endswith("." + t1_domain):
                return {
                    "score": HEALTH_AUTHORITY_SCORES["HEALTH_TIER_1"],
                    "tier": "HEALTH_TIER_1",
                    "domain": domain,
                    "reasoning": f"Health Tier 1 domain: {domain}",
                }

        # Check Tier 2
        for t2_domain in HEALTH_TIER_2_DOMAINS:
            if domain == t2_domain or domain.endswith("." + t2_domain):
                return {
                    "score": HEALTH_AUTHORITY_SCORES["HEALTH_TIER_2"],
                    "tier": "HEALTH_TIER_2",
                    "domain": domain,
                    "reasoning": f"Health Tier 2 domain: {domain}",
                }

        # Tier 2: .gov domains with health context (includes .gov.xx country variants)
        if (".gov" in domain) and domain not in HEALTH_TIER_1_DOMAINS:
            return {
                "score": HEALTH_AUTHORITY_SCORES["HEALTH_TIER_2"],
                "tier": "HEALTH_TIER_2",
                "domain": domain,
                "reasoning": f"Government health domain: {domain}",
            }

        # Check Tier 3 patterns
        for pattern in HEALTH_TIER_3_PATTERNS:
            if pattern in domain:
                return {
                    "score": HEALTH_AUTHORITY_SCORES["HEALTH_TIER_3"],
                    "tier": "HEALTH_TIER_3",
                    "domain": domain,
                    "reasoning": f"Health Tier 3 (pattern match: {pattern}): {domain}",
                }

        # Check Tier 3 specific domains
        for t3_domain in HEALTH_TIER_3_DOMAINS:
            if domain == t3_domain or domain.endswith("." + t3_domain):
                return {
                    "score": HEALTH_AUTHORITY_SCORES["HEALTH_TIER_3"],
                    "tier": "HEALTH_TIER_3",
                    "domain": domain,
                    "reasoning": f"Health Tier 3 domain: {domain}",
                }

        # Untiered
        return {
            "score": HEALTH_AUTHORITY_SCORES["HEALTH_UNTIERED"],
            "tier": "HEALTH_UNTIERED",
            "domain": domain,
            "reasoning": f"Untiered health domain: {domain}",
        }

    except Exception:
        return {
            "score": HEALTH_AUTHORITY_SCORES["HEALTH_UNTIERED"],
            "tier": "HEALTH_UNTIERED",
            "domain": "unknown",
            "reasoning": "Error parsing URL",
        }


def get_health_recency_score(
    publication_date: Optional[str] = None,
    current_date: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Calculate health-specific recency score with stricter decay.

    Health recency:
      < 2 years: 1.0
      2-5 years: 0.7
      > 5 years: 0.4
      No date:   0.3

    Returns:
        Dict with 'score' (0.0-1.0), 'years_old', 'category', 'reasoning'
    """
    if current_date is None:
        current_date = date.today()

    if not publication_date:
        return {
            "score": HEALTH_RECENCY["unknown"]["score"],
            "years_old": None,
            "category": "UNKNOWN",
            "reasoning": "No publication date - health penalty applied",
        }

    try:
        # Parse date — try common formats
        pub_date = _parse_date(publication_date)
        if not pub_date:
            return {
                "score": HEALTH_RECENCY["unknown"]["score"],
                "years_old": None,
                "category": "UNKNOWN",
                "reasoning": f"Unable to parse date: {publication_date}",
            }

        age_years = (current_date - pub_date).days / 365.25

        if age_years < HEALTH_RECENCY["recent"]["max_years"]:
            score = HEALTH_RECENCY["recent"]["score"]
            category = "RECENT"
        elif age_years <= HEALTH_RECENCY["moderate"]["max_years"]:
            score = HEALTH_RECENCY["moderate"]["score"]
            category = "MODERATE"
        else:
            score = HEALTH_RECENCY["outdated"]["score"]
            category = "OUTDATED"

        return {
            "score": score,
            "years_old": round(age_years, 1),
            "category": category,
            "reasoning": f"{age_years:.1f} years old - health recency: {category.lower()}",
        }

    except Exception:
        return {
            "score": HEALTH_RECENCY["unknown"]["score"],
            "years_old": None,
            "category": "UNKNOWN",
            "reasoning": "Error computing recency",
        }


def _parse_date(date_string: str) -> Optional[date]:
    """Parse date string in various formats."""
    import re
    from datetime import datetime

    date_string = date_string.strip()
    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
        "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt).date()
        except ValueError:
            continue

    # Regex fallback for year extraction
    year_match = re.search(r'\b(19|20)\d{2}\b', date_string)
    if year_match:
        try:
            return date(int(year_match.group()), 1, 1)
        except ValueError:
            pass

    return None
