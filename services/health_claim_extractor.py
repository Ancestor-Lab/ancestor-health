"""
Health Claim Extractor for Ancestor Health.
Decomposes AI-generated health text into discrete, verifiable factual claims.

Pipeline: sentence segmentation → claim pattern matching → compound claim splitting
          → self-containment rewrite → deduplication.

Rule-based only (spaCy + regex). No LLM in the extraction path. Deterministic.
Performance target: < 200ms for 20 claims from 2000-word input.
"""

import re
import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Extraction model version
EXTRACTION_MODEL = "ancestor-health-extractor-v1.0"

# Claim type patterns
CLAIM_PATTERNS = {
    "biomarker_interpretation": [
        # "[biomarker] + [indicates/suggests/associated with] + [condition]"
        re.compile(
            r"(?:elevated|high|low|decreased|increased|abnormal|normal)\s+.{3,40}\s+"
            r"(?:indicate[s]?|suggest[s]?|associated\s+with|linked\s+to|"
            r"may\s+(?:indicate|suggest|mean)|could\s+(?:indicate|suggest)|"
            r"is\s+(?:a\s+sign\s+of|indicative\s+of|consistent\s+with))"
            r".{5,}",
            re.IGNORECASE,
        ),
        # "[value] is above/below/within [range]"
        re.compile(
            r"\d+\.?\d*\s*(?:×\s*10[³²]?/?[µu]?L|g/dL|mg/dL|mmol/L|%|U/L|ng/mL|fL|pg)?\s*"
            r"(?:,\s*which\s+)?(?:is|are|was|falls?)\s+"
            r"(?:above|below|within|outside|higher\s+than|lower\s+than)",
            re.IGNORECASE,
        ),
    ],
    "normal_range_statement": [
        # "[value] falls within the normal range"
        re.compile(
            r"(?:normal|reference|expected)\s+range\s+(?:of|for|is)\s+\d",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:falls?|is|are)\s+(?:within|inside|in)\s+(?:the\s+)?normal\s+range",
            re.IGNORECASE,
        ),
    ],
    "condition_association": [
        re.compile(
            r"(?:this|these|such|the)\s+.{3,30}\s+"
            r"(?:elevation|level|count|result|finding|value)[s]?\s+"
            r"(?:may|could|might|can)\s+"
            r"(?:indicate|suggest|be\s+(?:a\s+sign|indicative|caused))",
            re.IGNORECASE,
        ),
    ],
    "treatment_statement": [
        # "[condition] + [treated with/managed by] + [treatment]"
        re.compile(
            r"(?:treat(?:ed|ment)?|manag(?:ed|ement)|therap(?:y|ies)|prescri(?:be|bed))\s+"
            r"(?:with|by|using|include[s]?)\s+.{3,}",
            re.IGNORECASE,
        ),
    ],
    "prevalence_stat": [
        re.compile(
            r"(?:affect[s]?|occur[s]?\s+in|prevalence|incidence)\s+"
            r"(?:approximately|about|roughly|nearly)?\s*\d",
            re.IGNORECASE,
        ),
    ],
    "guideline_reference": [
        re.compile(
            r"(?:WHO|CDC|NIH|AHA|ACC|NICE|guideline[s]?|recommend(?:s|ation)?|advis(?:es|ory))\s+"
            r"(?:recommend[s]?|suggest[s]?|advis(?:es|e)|state[s]?)",
            re.IGNORECASE,
        ),
    ],
    "mechanism_statement": [
        re.compile(
            r"(?:caused?\s+by|results?\s+from|due\s+to|leads?\s+to|"
            r"contributes?\s+to|associated\s+with|risk\s+factor\s+for)",
            re.IGNORECASE,
        ),
    ],
}

# Compound claim splitting conjunctions
COMPOUND_SPLITTERS = re.compile(
    r"\s+(?:and|or|as\s+well\s+as|in\s+addition\s+to)\s+",
    re.IGNORECASE,
)

# Pronouns to resolve for self-containment
PRONOUN_PATTERNS = re.compile(
    r"\b(this|it|these|those|that|they|its|their|such)\b",
    re.IGNORECASE,
)


@dataclass
class ExtractedClaim:
    """A discrete, verifiable factual claim extracted from health text."""
    claim_id: str
    claim_text: str
    original_span: tuple  # (start_char, end_char)
    claim_type: str
    confidence: float  # 0.0–1.0


@dataclass
class ClaimExtractionResult:
    """Result of claim extraction from health text."""
    claims: list[ExtractedClaim]
    input_text: str
    input_type: str
    extraction_model: str
    extraction_latency_ms: int


class HealthClaimExtractor:
    """
    Decomposes health text into verifiable claims.
    Does NOT generate new content. Does NOT interpret.
    Extraction only. Rule-based, deterministic.
    """

    def __init__(self):
        self._nlp = None

    def _get_nlp(self):
        """Lazy-load spaCy model for sentence segmentation."""
        if self._nlp is None:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # Fallback: use blank English model with sentencizer
                    self._nlp = spacy.blank("en")
                    self._nlp.add_pipe("sentencizer")
            except ImportError:
                logger.warning("spaCy not available, using regex sentence splitting")
                self._nlp = "regex_fallback"
        return self._nlp

    async def extract(
        self,
        text: str,
        input_type: str,
        context: Optional[str] = None,
        max_claims: int = 20,
    ) -> ClaimExtractionResult:
        """
        Extract verifiable health claims from AI-generated text.

        Args:
            text: AI-generated health text.
            input_type: Type of input (ai_health_response, lab_result_explanation, etc.)
            context: Optional clinical context for self-containment rewrites.
            max_claims: Maximum claims to extract.

        Returns:
            ClaimExtractionResult with extracted claims.
        """
        start_time = time.monotonic()

        # Step 0: Preprocess lab explanations
        biomarker_context = {}
        if input_type == "lab_result_explanation":
            from .lab_explanation_handler import LabExplanationHandler
            handler = LabExplanationHandler()
            annotated = handler.preprocess(text)
            # Build context map from biomarker references
            for ref in annotated.biomarkers:
                biomarker_context[ref.span] = ref

        # Step 1: Sentence segmentation
        sentences = self._segment_sentences(text)

        # Step 2: Claim pattern matching
        raw_claims = []
        for sent_text, sent_start, sent_end in sentences:
            claim_type = self._classify_sentence(sent_text)
            if claim_type:
                raw_claims.append((sent_text, sent_start, sent_end, claim_type))

        # Step 3: Compound claim splitting
        split_claims = []
        for claim_text, start, end, claim_type in raw_claims:
            splits = self._split_compound_claims(claim_text, start, claim_type)
            split_claims.extend(splits)

        # Step 4: Self-containment rewrite
        rewritten = []
        for claim_text, start, end, claim_type in split_claims:
            rewritten_text = self._ensure_self_contained(
                claim_text, text, start, context, biomarker_context
            )
            rewritten.append((rewritten_text, start, end, claim_type))

        # Step 5: Deduplication
        unique_claims = self._deduplicate(rewritten)

        # Step 6: Build result with confidence scoring
        claims = []
        for i, (claim_text, start, end, claim_type) in enumerate(unique_claims[:max_claims]):
            confidence = self._compute_confidence(claim_text, claim_type)
            claims.append(
                ExtractedClaim(
                    claim_id=str(uuid.uuid4()),
                    claim_text=claim_text.strip(),
                    original_span=(start, end),
                    claim_type=claim_type,
                    confidence=confidence,
                )
            )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return ClaimExtractionResult(
            claims=claims,
            input_text=text,
            input_type=input_type,
            extraction_model=EXTRACTION_MODEL,
            extraction_latency_ms=elapsed_ms,
        )

    def _segment_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences with character offsets."""
        nlp = self._get_nlp()

        if nlp == "regex_fallback":
            return self._regex_sentence_split(text)

        doc = nlp(text)
        sentences = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) > 10:  # Skip trivial fragments
                sentences.append((sent_text, sent.start_char, sent.end_char))
        return sentences

    def _regex_sentence_split(self, text: str) -> list[tuple[str, int, int]]:
        """Fallback sentence splitting using regex.
        Handles decimal numbers (e.g., 15.2) to avoid mid-number splits.
        """
        sentences = []
        # Split on period/!/?  followed by space+uppercase or end of string,
        # but NOT on decimal points (digit.digit)
        pattern = re.compile(
            r'(?<!\d)(?<!\d\.\d)[.!?]\s+(?=[A-Z])|'  # sentence boundary
            r'(?<!\d)(?<!\d\.\d)[.!?]\s*$'             # end of text
        )

        last = 0
        for match in pattern.finditer(text):
            end = match.end()
            sent = text[last:end].strip()
            if len(sent) > 10:
                sentences.append((sent, last, last + len(sent)))
            last = end

        # Capture trailing text
        if last < len(text):
            sent = text[last:].strip()
            if len(sent) > 10:
                sentences.append((sent, last, last + len(sent)))

        return sentences

    def _classify_sentence(self, sentence: str) -> Optional[str]:
        """
        Match a sentence against health claim patterns.
        Returns claim type or None if no match.
        """
        # Check each claim type's patterns
        for claim_type, patterns in CLAIM_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(sentence):
                    return claim_type

        # Heuristic: sentences with numbers + medical context are likely claims
        has_number = bool(re.search(r'\d+\.?\d*\s*(?:×|g/|mg/|mmol|%|/[µu]L)', sentence))
        has_medical = bool(re.search(
            r'(?:blood|cell|count|level|range|normal|abnormal|elevated|'
            r'infection|disease|treatment|diagnosis|symptom|risk|'
            r'anemia|diabetes|cholesterol|pressure|heart|liver|kidney)',
            sentence, re.IGNORECASE
        ))
        if has_number and has_medical:
            return "biomarker_interpretation"

        # Sentences with health assessment language
        if has_medical and re.search(
            r'(?:may|could|might|suggest|indicate|associated|caused|risk|'
            r'recommend|should|important|significant)',
            sentence, re.IGNORECASE
        ):
            return "condition_association"

        return None

    def _split_compound_claims(
        self, text: str, start: int, claim_type: str
    ) -> list[tuple[str, int, int, str]]:
        """
        Split compound claims containing 'and'/'or' into atomic claims.
        Only splits when both halves look like separate claims.
        """
        # Don't split short sentences
        if len(text) < 60:
            return [(text, start, start + len(text), claim_type)]

        # Try splitting on "or" for condition alternatives
        parts = re.split(r'\s+or\s+', text, maxsplit=1)
        if len(parts) == 2 and len(parts[0]) > 20 and len(parts[1]) > 15:
            # Check if the second part needs the subject from the first
            if not parts[1][0].isupper() and not re.match(r'\d', parts[1]):
                # Carry forward the subject
                subject_match = re.match(r'^(.+?\s+(?:may|could|might|can)\s+)', parts[0])
                if subject_match:
                    prefix = subject_match.group(1)
                    parts[1] = prefix + parts[1]

            results = []
            offset = start
            for part in parts:
                part = part.strip()
                if len(part) > 15:
                    results.append((part, offset, offset + len(part), claim_type))
                offset += len(part) + 4  # " or "
            if results:
                return results

        return [(text, start, start + len(text), claim_type)]

    def _ensure_self_contained(
        self,
        claim_text: str,
        full_text: str,
        claim_start: int,
        context: Optional[str],
        biomarker_context: dict,
    ) -> str:
        """
        Rewrite claim to be self-contained if it references pronouns/implicit subjects.
        Uses surrounding context, not generation.
        """
        # Check for pronouns that need resolution
        if not PRONOUN_PATTERNS.search(claim_text):
            return claim_text

        # Find the preceding sentence for context
        preceding_text = full_text[:claim_start].strip()
        preceding_sentences = re.split(r'[.!?]\s+', preceding_text)
        prev_sentence = preceding_sentences[-1] if preceding_sentences else ""

        # Try to find a subject in the preceding sentence
        subject = self._extract_subject(prev_sentence)
        if not subject:
            # Try biomarker context
            for span, ref in biomarker_context.items():
                if span[1] <= claim_start:
                    subject = ref.biomarker_name
                    break

        if subject:
            # Replace pronouns with the identified subject
            result = claim_text
            for pronoun in ["This", "this", "It", "it", "These", "these", "That", "that"]:
                if result.startswith(pronoun + " "):
                    result = subject + " " + result[len(pronoun) + 1:]
                    break
                result = re.sub(
                    rf'\b{pronoun}\b(?!\s+(?:is|are|was|were|may|could|might|can|will|should))',
                    subject,
                    result,
                    count=1,
                )
            return result

        return claim_text

    def _extract_subject(self, sentence: str) -> Optional[str]:
        """Extract the likely subject from a sentence."""
        if not sentence:
            return None

        # Look for biomarker/medical subjects
        patterns = [
            r"(?:your\s+)?(\w+(?:\s+\w+){0,3}\s+(?:count|level|value|result|ratio|rate))",
            r"(?:the\s+)?(\w+(?:\s+\w+){0,2}\s+(?:test|measurement|reading))",
            r"((?:white|red)\s+blood\s+cell\s+count)",
            r"(hemoglobin(?:\s+level)?)",
            r"(platelet\s+count)",
            r"(blood\s+(?:sugar|glucose|pressure))",
        ]

        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _deduplicate(
        self, claims: list[tuple[str, int, int, str]]
    ) -> list[tuple[str, int, int, str]]:
        """Remove semantically duplicate claims using token overlap."""
        if not claims:
            return claims

        unique = [claims[0]]
        for claim in claims[1:]:
            is_dup = False
            claim_tokens = set(claim[0].lower().split())
            for existing in unique:
                existing_tokens = set(existing[0].lower().split())
                if claim_tokens and existing_tokens:
                    overlap = len(claim_tokens & existing_tokens) / len(
                        claim_tokens | existing_tokens
                    )
                    if overlap > 0.8:  # 80% token overlap = duplicate
                        is_dup = True
                        break
            if not is_dup:
                unique.append(claim)

        return unique

    def _compute_confidence(self, claim_text: str, claim_type: str) -> float:
        """
        Compute extraction confidence score (0.0–1.0).
        Higher for claims with specific numbers, known patterns, clear structure.
        """
        confidence = 0.5  # base

        # Boost for specific numeric values
        if re.search(r'\d+\.?\d*\s*(?:×|g/|mg/|mmol|%|/[µu]L|U/L)', claim_text):
            confidence += 0.2

        # Boost for explicit medical terms
        medical_terms = re.findall(
            r'(?:infection|anemia|diabetes|cholesterol|pressure|inflammatory|'
            r'bacterial|viral|normal\s+range|elevated|decreased)',
            claim_text, re.IGNORECASE
        )
        if medical_terms:
            confidence += min(0.15, len(medical_terms) * 0.05)

        # Boost for clear claim structure
        if re.search(
            r'(?:indicate|suggest|associated|caused|risk|recommend|normal|abnormal)',
            claim_text, re.IGNORECASE
        ):
            confidence += 0.1

        # Penalize very short or very long claims
        word_count = len(claim_text.split())
        if word_count < 5:
            confidence -= 0.2
        elif word_count > 50:
            confidence -= 0.1

        return round(max(0.1, min(1.0, confidence)), 2)
