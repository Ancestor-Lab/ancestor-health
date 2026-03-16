"""
Phase 4A: Source Integrity Protocol for Weighted Trust Assessment
Calculates Provenance, Recency, and Integrity scores for composite Trust Score calculation.

This module implements the enterprise-grade integrity dimensions:
- Provenance (30%): Domain authority and credibility
- Recency (20%): Content freshness based on publication date
- Integrity (10%): Basic quality and spam detection
"""

import re
import logging
from datetime import datetime, date
from typing import Dict, Optional, Any
from urllib.parse import urlparse

from .health_scoring_config import get_health_authority_score, get_health_recency_score

logger = logging.getLogger(__name__)

# Phase 4A: Scoring Weight Constants
PROVENANCE_WEIGHT = 0.30
RECENCY_WEIGHT = 0.20
INTEGRITY_WEIGHT = 0.10
ENTAILMENT_WEIGHT = 0.40

# Provenance Scoring Constants
TIER_1_SCORE = 100
NEUTRAL_SCORE = 50

# Recency Scoring Constants
RECENT_SCORE = 100  # < 2 years
MODERATE_SCORE = 80  # 2-5 years
OUTDATED_SCORE = 50  # > 5 years
UNKNOWN_DATE_SCORE = 50

# Integrity Scoring Constants
MIN_CONTENT_LENGTH = 500
INTEGRITY_MAX_SCORE = 100

class HeuristicScorer:
    """
    Phase 4A: Source Integrity Protocol for Trust Score calculation.
    Implements Provenance, Recency, and Integrity scoring dimensions.
    """

    def __init__(self):
        """Initialize the heuristic scorer with domain allowlists and patterns."""
        # Tier 1 Domains - Highest Authority (Score: 100)
        self.tier_1_domains = {
            # Government Health Organizations
            'who.int', 'cdc.gov', 'nih.gov', 'fda.gov',
            # Academic/Research
            'nature.com', 'science.org', 'sciencemag.org',
            # Trusted News
            'reuters.com', 'ap.org', 'apnews.com',
            # Medical Authorities
            'nejm.org', 'thelancet.com', 'bmj.com',
            # International Organizations
            'un.org', 'unicef.org'
        }

        # TLD Authority Mapping
        self.authority_tlds = {
            '.gov': TIER_1_SCORE,
            '.edu': TIER_1_SCORE,
            '.mil': TIER_1_SCORE
        }

        logger.info(f"HeuristicScorer initialized with {len(self.tier_1_domains)} Tier 1 domains")

    def calculate_provenance_score(self, source_url: str, scoring_profile: str = "general") -> Dict[str, Any]:
        """
        Calculate Provenance score based on domain authority.

        Args:
            source_url: Source URL to score
            scoring_profile: "general" (0-100 scale) or "health" (0.0-1.0 scale)

        Returns:
            Dict with score and reasoning
        """
        if scoring_profile == "health":
            return get_health_authority_score(source_url)

        try:
            parsed_url = urlparse(source_url.lower())
            domain = parsed_url.netloc

            # Remove 'www.' prefix for domain matching
            if domain.startswith('www.'):
                clean_domain = domain[4:]
            else:
                clean_domain = domain

            # Check Tier 1 domain allowlist
            if clean_domain in self.tier_1_domains:
                return {
                    'score': TIER_1_SCORE,
                    'tier': 'TIER_1_ALLOWLIST',
                    'domain': clean_domain,
                    'reasoning': f'Tier 1 authority domain: {clean_domain}'
                }

            # Check TLD authority (.gov, .edu, .mil)
            for tld, score in self.authority_tlds.items():
                if clean_domain.endswith(tld):
                    return {
                        'score': score,
                        'tier': 'TIER_1_TLD',
                        'domain': clean_domain,
                        'reasoning': f'Authority TLD: {tld}'
                    }

            # Default: Unknown domain = Neutral
            return {
                'score': NEUTRAL_SCORE,
                'tier': 'UNKNOWN_NEUTRAL',
                'domain': clean_domain,
                'reasoning': 'Unknown domain - neutral scoring'
            }

        except Exception as e:
            logger.warning(f"Provenance scoring failed for {source_url}: {str(e)}")
            return {
                'score': NEUTRAL_SCORE,
                'tier': 'ERROR_NEUTRAL',
                'domain': 'unknown',
                'reasoning': f'Error parsing URL: {str(e)}'
            }

    def calculate_recency_score(self, publication_date: Optional[str] = None,
                              current_date: Optional[date] = None,
                              scoring_profile: str = "general") -> Dict[str, Any]:
        """
        Calculate Recency score based on publication date decay function.

        Args:
            publication_date: Publication date string (various formats)
            current_date: Current date for comparison (defaults to today)
            scoring_profile: "general" (0-100 scale) or "health" (0.0-1.0 scale)

        Returns:
            Dict with score and reasoning
        """
        if scoring_profile == "health":
            return get_health_recency_score(publication_date, current_date)

        if current_date is None:
            current_date = date.today()

        if not publication_date:
            return {
                'score': UNKNOWN_DATE_SCORE,
                'years_old': None,
                'reasoning': 'No publication date provided - neutral scoring'
            }

        try:
            # Parse publication date - handle multiple formats
            pub_date = self._parse_date(publication_date)
            if not pub_date:
                return {
                    'score': UNKNOWN_DATE_SCORE,
                    'years_old': None,
                    'reasoning': f'Unable to parse date: {publication_date}'
                }

            # Calculate age in years
            age_years = (current_date - pub_date).days / 365.25

            # Apply decay function
            if age_years < 2:
                score = RECENT_SCORE
                category = 'RECENT'
            elif age_years <= 5:
                score = MODERATE_SCORE
                category = 'MODERATE'
            else:
                score = OUTDATED_SCORE
                category = 'OUTDATED'

            return {
                'score': score,
                'years_old': round(age_years, 1),
                'category': category,
                'reasoning': f'{age_years:.1f} years old - {category.lower()} content'
            }

        except Exception as e:
            logger.warning(f"Recency scoring failed for {publication_date}: {str(e)}")
            return {
                'score': UNKNOWN_DATE_SCORE,
                'years_old': None,
                'reasoning': f'Error parsing date: {str(e)}'
            }

    def calculate_integrity_score(self, content: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate Integrity score (0-100) based on basic quality checks.
        Acts as a spam filter, not a credibility booster.

        Args:
            content: Text content to analyze
            title: Optional title for additional checks

        Returns:
            Dict with score and quality indicators
        """
        if not content:
            return {
                'score': 0,
                'checks': {'content_length': False},
                'reasoning': 'No content provided'
            }

        checks = {}
        score_components = []

        # Check 1: Content Length (Primary spam filter)
        content_length_check = len(content.strip()) >= MIN_CONTENT_LENGTH
        checks['content_length'] = content_length_check
        if content_length_check:
            score_components.append(40)  # 40% for adequate length

        # Check 2: Proper Capitalization (Basic quality indicator)
        capitalization_check = self._check_capitalization(content)
        checks['proper_capitalization'] = capitalization_check
        if capitalization_check:
            score_components.append(30)  # 30% for proper formatting

        # Check 3: Sentence Structure (Basic coherence check)
        sentence_structure_check = self._check_sentence_structure(content)
        checks['sentence_structure'] = sentence_structure_check
        if sentence_structure_check:
            score_components.append(30)  # 30% for coherent structure

        # Calculate final integrity score
        final_score = sum(score_components)

        # Determine quality category
        if final_score >= 80:
            category = 'HIGH_QUALITY'
        elif final_score >= 50:
            category = 'MODERATE_QUALITY'
        else:
            category = 'LOW_QUALITY'

        return {
            'score': final_score,
            'category': category,
            'checks': checks,
            'content_length': len(content.strip()),
            'reasoning': f'{category.replace("_", " ").lower()} content - {len(score_components)}/3 checks passed'
        }

    def _parse_date(self, date_string: str) -> Optional[date]:
        """
        Parse various date formats to extract publication date.

        Args:
            date_string: Date string in various formats

        Returns:
            Parsed date object or None if parsing fails
        """
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d',          # 2024-01-15
            '%Y/%m/%d',          # 2024/01/15
            '%d/%m/%Y',          # 15/01/2024
            '%m/%d/%Y',          # 01/15/2024
            '%B %d, %Y',         # January 15, 2024
            '%b %d, %Y',         # Jan 15, 2024
            '%d %B %Y',          # 15 January 2024
            '%d %b %Y',          # 15 Jan 2024
            '%Y',                # 2024 (year only)
        ]

        date_string = date_string.strip()

        # Try each format
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_string, fmt).date()
                return parsed_date
            except ValueError:
                continue

        # Try to extract year with regex if standard formats fail
        year_match = re.search(r'\b(19|20)\d{2}\b', date_string)
        if year_match:
            try:
                year = int(year_match.group())
                return date(year, 1, 1)  # Use January 1st for year-only dates
            except ValueError:
                pass

        return None

    def _check_capitalization(self, content: str) -> bool:
        """
        Check if content has proper capitalization patterns.

        Args:
            content: Text content to check

        Returns:
            True if capitalization appears proper
        """
        if not content:
            return False

        # Split into sentences
        sentences = re.split(r'[.!?]+', content.strip())
        if len(sentences) < 2:
            return False

        # Check if most sentences start with capital letters
        capitalized_sentences = 0
        total_sentences = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 3:  # Skip very short fragments
                total_sentences += 1
                if sentence[0].isupper():
                    capitalized_sentences += 1

        if total_sentences == 0:
            return False

        # At least 70% of sentences should start with capitals
        capitalization_ratio = capitalized_sentences / total_sentences
        return capitalization_ratio >= 0.7

    def _check_sentence_structure(self, content: str) -> bool:
        """
        Check basic sentence structure quality.

        Args:
            content: Text content to check

        Returns:
            True if sentence structure appears coherent
        """
        if not content:
            return False

        # Split into sentences
        sentences = re.split(r'[.!?]+', content.strip())
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(valid_sentences) < 3:
            return False

        # Check average sentence length (spam often has very short or very long sentences)
        total_words = sum(len(sentence.split()) for sentence in valid_sentences)
        avg_sentence_length = total_words / len(valid_sentences)

        # Reasonable sentence length: 8-40 words
        return 8 <= avg_sentence_length <= 40

    def get_scoring_weights(self) -> Dict[str, float]:
        """
        Get the scoring weights for all dimensions.

        Returns:
            Dict with dimension weights
        """
        return {
            'entailment': ENTAILMENT_WEIGHT,
            'provenance': PROVENANCE_WEIGHT,
            'recency': RECENCY_WEIGHT,
            'integrity': INTEGRITY_WEIGHT
        }
