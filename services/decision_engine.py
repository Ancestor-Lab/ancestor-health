"""
Phase 4A: Decision Engine with Weighted Heuristic Scoring
Implements composite Trust Score calculation using four weighted dimensions:
- Entailment (40%): NLI semantic analysis
- Provenance (30%): Domain authority and credibility
- Recency (20%): Content freshness decay
- Integrity (10%): Basic quality and spam detection

The Invariant: IF Contradiction > 0.5 THEN Trust Score = 0 (Kill Switch)
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

from .heuristic_scorer import HeuristicScorer
from .nli_service import NLIService

logger = logging.getLogger(__name__)

# Phase 4A: Scoring Weight Constants (Frozen)
ENTAILMENT_WEIGHT = 0.40
PROVENANCE_WEIGHT = 0.30
RECENCY_WEIGHT = 0.20
INTEGRITY_WEIGHT = 0.10

# Trust Score Thresholds
HIGH_TRUST_THRESHOLD = 80
MODERATE_TRUST_THRESHOLD = 50

# Kill Switch Threshold
KILL_SWITCH_THRESHOLD = 0.5

class DecisionEngine:
    """
    Phase 4A: Weighted Heuristic Decision Engine for Trust Score calculation.
    Combines NLI analysis with domain authority, recency, and integrity scoring.
    """

    def __init__(self, nli_service: Optional[NLIService] = None):
        """
        Initialize the Decision Engine.

        Args:
            nli_service: Optional NLI service instance (will create if not provided)
        """
        self.nli_service = nli_service
        self.heuristic_scorer = HeuristicScorer()

        # Scoring weights (frozen constants)
        self.weights = {
            'entailment': ENTAILMENT_WEIGHT,
            'provenance': PROVENANCE_WEIGHT,
            'recency': RECENCY_WEIGHT,
            'integrity': INTEGRITY_WEIGHT
        }

        logger.info("DecisionEngine initialized with weighted heuristic scoring")

    def calculate_trust_score(self,
                            claim: str,
                            evidence: str,
                            source_url: str,
                            publication_date: Optional[str] = None,
                            title: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate composite Trust Score using weighted heuristic dimensions.

        Args:
            claim: The claim to verify
            evidence: Evidence text from source
            source_url: Source URL for provenance scoring
            publication_date: Publication date for recency scoring
            title: Optional title for integrity scoring

        Returns:
            Complete trust assessment with component breakdown
        """
        try:
            # Step 1: NLI Analysis (Entailment dimension)
            if not self.nli_service:
                # Lazy load NLI service to avoid startup delays
                from .nli_service import NLIService
                self.nli_service = NLIService()

            nli_result = self.nli_service.analyze_claim_evidence(claim, evidence)
            nli_probabilities = nli_result['raw_probabilities']

            # Step 2: Apply Kill Switch (The Invariant)
            contradiction_prob = nli_probabilities.get('contradiction', 0.0)
            if contradiction_prob > KILL_SWITCH_THRESHOLD:
                return self._create_kill_switch_response(claim, source_url, nli_result, contradiction_prob)

            # Step 3: Calculate component scores
            entailment_prob = nli_probabilities.get('entailment', 0.0)
            entailment_score = entailment_prob * 100  # Linear mapping: 1.0 -> 100

            # Provenance scoring
            provenance_result = self.heuristic_scorer.calculate_provenance_score(source_url)
            provenance_score = provenance_result['score']

            # Recency scoring
            recency_result = self.heuristic_scorer.calculate_recency_score(publication_date)
            recency_score = recency_result['score']

            # Integrity scoring
            integrity_result = self.heuristic_scorer.calculate_integrity_score(evidence, title)
            integrity_score = integrity_result['score']

            # Step 4: Calculate weighted composite score
            final_score = (
                (entailment_score * self.weights['entailment']) +
                (provenance_score * self.weights['provenance']) +
                (recency_score * self.weights['recency']) +
                (integrity_score * self.weights['integrity'])
            )

            # Step 5: Apply labeling rules
            trust_label = self._determine_trust_label(final_score)

            # Step 6: Create comprehensive response
            response = {
                'claim': claim,
                'source_url': source_url,
                'final_score': round(final_score, 2),
                'trust_label': trust_label,
                'kill_switch_triggered': False,
                'component_scores': {
                    'entailment': {
                        'score': round(entailment_score, 2),
                        'weight': self.weights['entailment'],
                        'weighted_contribution': round(entailment_score * self.weights['entailment'], 2),
                        'nli_probability': entailment_prob,
                        'nli_details': nli_result
                    },
                    'provenance': {
                        'score': provenance_score,
                        'weight': self.weights['provenance'],
                        'weighted_contribution': round(provenance_score * self.weights['provenance'], 2),
                        'details': provenance_result
                    },
                    'recency': {
                        'score': recency_score,
                        'weight': self.weights['recency'],
                        'weighted_contribution': round(recency_score * self.weights['recency'], 2),
                        'details': recency_result
                    },
                    'integrity': {
                        'score': integrity_score,
                        'weight': self.weights['integrity'],
                        'weighted_contribution': round(integrity_score * self.weights['integrity'], 2),
                        'details': integrity_result
                    }
                },
                'scoring_formula': {
                    'formula': f"({entailment_score:.1f} × {self.weights['entailment']}) + ({provenance_score} × {self.weights['provenance']}) + ({recency_score} × {self.weights['recency']}) + ({integrity_score} × {self.weights['integrity']}) = {final_score:.2f}",
                    'weights': self.weights,
                    'thresholds': {
                        'high_trust': HIGH_TRUST_THRESHOLD,
                        'moderate_trust': MODERATE_TRUST_THRESHOLD,
                        'kill_switch': KILL_SWITCH_THRESHOLD
                    }
                },
                'phase': 'Phase_4A_Weighted_Heuristic'
            }

            logger.info(f"Trust score calculated: {final_score:.2f} ({trust_label}) for {source_url}")
            return response

        except Exception as e:
            logger.error(f"Trust score calculation failed: {str(e)}")
            return self._create_error_response(claim, source_url, str(e))

    def _create_kill_switch_response(self,
                                   claim: str,
                                   source_url: str,
                                   nli_result: Dict[str, Any],
                                   contradiction_prob: float) -> Dict[str, Any]:
        """
        Create response when kill switch is triggered by high contradiction.

        Args:
            claim: The original claim
            source_url: Source URL
            nli_result: NLI analysis result
            contradiction_prob: Contradiction probability that triggered kill switch

        Returns:
            Kill switch response with zero trust score
        """
        return {
            'claim': claim,
            'source_url': source_url,
            'final_score': 0.0,
            'trust_label': 'BLOCKED_CONTRADICTION',
            'kill_switch_triggered': True,
            'kill_switch_reason': f'Contradiction probability {contradiction_prob:.3f} > {KILL_SWITCH_THRESHOLD}',
            'component_scores': {
                'entailment': {
                    'score': 0.0,
                    'weight': self.weights['entailment'],
                    'weighted_contribution': 0.0,
                    'nli_probability': nli_result['raw_probabilities'].get('entailment', 0.0),
                    'nli_details': nli_result
                },
                'provenance': {'score': 'N/A', 'weight': self.weights['provenance'], 'weighted_contribution': 0.0},
                'recency': {'score': 'N/A', 'weight': self.weights['recency'], 'weighted_contribution': 0.0},
                'integrity': {'score': 'N/A', 'weight': self.weights['integrity'], 'weighted_contribution': 0.0}
            },
            'scoring_formula': {
                'formula': 'KILL SWITCH OVERRIDES ALL WEIGHTS - FINAL SCORE = 0',
                'weights': self.weights,
                'kill_switch_triggered': True,
                'thresholds': {
                    'high_trust': HIGH_TRUST_THRESHOLD,
                    'moderate_trust': MODERATE_TRUST_THRESHOLD,
                    'kill_switch': KILL_SWITCH_THRESHOLD
                }
            },
            'phase': 'Phase_4A_Weighted_Heuristic'
        }

    def _create_error_response(self, claim: str, source_url: str, error_msg: str) -> Dict[str, Any]:
        """
        Create error response when trust score calculation fails.

        Args:
            claim: The original claim
            source_url: Source URL
            error_msg: Error message

        Returns:
            Error response
        """
        return {
            'claim': claim,
            'source_url': source_url,
            'final_score': 0.0,
            'trust_label': 'ERROR',
            'kill_switch_triggered': False,
            'error_message': error_msg,
            'component_scores': {
                'entailment': {'score': 0.0, 'error': error_msg},
                'provenance': {'score': 0.0, 'error': error_msg},
                'recency': {'score': 0.0, 'error': error_msg},
                'integrity': {'score': 0.0, 'error': error_msg}
            },
            'phase': 'Phase_4A_Weighted_Heuristic'
        }

    def _determine_trust_label(self, score: float) -> str:
        """
        Determine trust label based on composite score.

        Args:
            score: Composite trust score (0-100)

        Returns:
            Trust label string
        """
        if score >= HIGH_TRUST_THRESHOLD:
            return 'HIGH_TRUST'
        elif score >= MODERATE_TRUST_THRESHOLD:
            return 'MODERATE_TRUST'
        else:
            return 'LOW_TRUST'

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the decision engine configuration.

        Returns:
            Engine configuration details
        """
        return {
            'phase': 'Phase_4A_Weighted_Heuristic',
            'weights': self.weights,
            'thresholds': {
                'high_trust': HIGH_TRUST_THRESHOLD,
                'moderate_trust': MODERATE_TRUST_THRESHOLD,
                'kill_switch': KILL_SWITCH_THRESHOLD
            },
            'dimensions': {
                'entailment': 'NLI semantic analysis (linear: probability × 100)',
                'provenance': 'Domain authority (.gov/.edu=100, Tier1=100, Unknown=50)',
                'recency': 'Content freshness (<2yr=100, 2-5yr=80, >5yr=50)',
                'integrity': 'Basic quality checks (length, capitalization, structure)'
            },
            'formula': 'final_score = (entailment × 0.40) + (provenance × 0.30) + (recency × 0.20) + (integrity × 0.10)',
            'invariant': 'IF contradiction > 0.5 THEN final_score = 0 (overrides all weights)',
            'heuristic_scorer': self.heuristic_scorer.__class__.__name__,
            'nli_service': self.nli_service.__class__.__name__ if self.nli_service else 'Lazy-loaded'
        }

    def validate_scoring_formula(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate scoring formula with a test case for debugging.

        Args:
            test_case: Test case with expected values

        Returns:
            Validation result
        """
        # This is primarily for unit testing
        expected_final = (
            test_case['entailment_score'] * ENTAILMENT_WEIGHT +
            test_case['provenance_score'] * PROVENANCE_WEIGHT +
            test_case['recency_score'] * RECENCY_WEIGHT +
            test_case['integrity_score'] * INTEGRITY_WEIGHT
        )

        return {
            'expected_score': round(expected_final, 2),
            'formula_breakdown': {
                'entailment': f"{test_case['entailment_score']} × {ENTAILMENT_WEIGHT} = {test_case['entailment_score'] * ENTAILMENT_WEIGHT:.2f}",
                'provenance': f"{test_case['provenance_score']} × {PROVENANCE_WEIGHT} = {test_case['provenance_score'] * PROVENANCE_WEIGHT:.2f}",
                'recency': f"{test_case['recency_score']} × {RECENCY_WEIGHT} = {test_case['recency_score'] * RECENCY_WEIGHT:.2f}",
                'integrity': f"{test_case['integrity_score']} × {INTEGRITY_WEIGHT} = {test_case['integrity_score'] * INTEGRITY_WEIGHT:.2f}"
            },
            'weights_total': sum(self.weights.values())
        }
