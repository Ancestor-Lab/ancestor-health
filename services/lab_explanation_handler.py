"""
Lab Result Explanation Handler for Ancestor Health.
Preprocessing for input_type: "lab_result_explanation".

Identifies biomarker references (name, value, unit, range) in AI-generated
lab result explanations using regex. Tags them so the claim extractor can
produce precise, verifiable claims.

Does NOT interpret lab data. The AI already did that — we verify its claims.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Common biomarker names and their aliases
BIOMARKER_ALIASES = {
    "wbc": "white blood cell count",
    "rbc": "red blood cell count",
    "hgb": "hemoglobin",
    "hb": "hemoglobin",
    "hct": "hematocrit",
    "plt": "platelet count",
    "mcv": "mean corpuscular volume",
    "mch": "mean corpuscular hemoglobin",
    "mchc": "mean corpuscular hemoglobin concentration",
    "rdw": "red cell distribution width",
    "mpv": "mean platelet volume",
    "ldl": "LDL cholesterol",
    "hdl": "HDL cholesterol",
    "ast": "aspartate aminotransferase",
    "alt": "alanine aminotransferase",
    "alp": "alkaline phosphatase",
    "ggt": "gamma-glutamyl transferase",
    "bun": "blood urea nitrogen",
    "egfr": "estimated glomerular filtration rate",
    "crp": "C-reactive protein",
    "esr": "erythrocyte sedimentation rate",
    "tsh": "thyroid-stimulating hormone",
    "hba1c": "hemoglobin A1c",
    "a1c": "hemoglobin A1c",
    "psa": "prostate-specific antigen",
    "bnp": "B-type natriuretic peptide",
    "inr": "international normalized ratio",
    "ptt": "partial thromboplastin time",
    "pt": "prothrombin time",
    "ferritin": "ferritin",
    "creatinine": "creatinine",
    "albumin": "albumin",
    "bilirubin": "bilirubin",
    "glucose": "glucose",
    "sodium": "sodium",
    "potassium": "potassium",
    "calcium": "calcium",
    "magnesium": "magnesium",
    "phosphate": "phosphate",
    "chloride": "chloride",
    "triglycerides": "triglycerides",
}

# Known biomarker name patterns (case-insensitive)
BIOMARKER_PATTERNS = [
    # Full names
    r"white\s+blood\s+cell\s+count",
    r"red\s+blood\s+cell\s+count",
    r"hemoglobin(?:\s+level)?",
    r"haemoglobin(?:\s+level)?",
    r"hematocrit",
    r"platelet\s+count",
    r"blood\s+(?:urea\s+)?nitrogen",
    r"creatinine(?:\s+level)?",
    r"(?:total\s+)?cholesterol",
    r"LDL(?:\s+cholesterol)?",
    r"HDL(?:\s+cholesterol)?",
    r"triglyceride[s]?",
    r"(?:fasting\s+)?(?:blood\s+)?glucose",
    r"hemoglobin\s+A1c",
    r"HbA1c",
    r"thyroid[- ]stimulating\s+hormone",
    r"C[- ]reactive\s+protein",
    r"erythrocyte\s+sedimentation\s+rate",
    r"ferritin(?:\s+level)?",
    r"albumin(?:\s+level)?",
    r"bilirubin(?:\s+level)?",
    r"alkaline\s+phosphatase",
    r"alanine\s+(?:amino)?transferase",
    r"aspartate\s+(?:amino)?transferase",
    # Abbreviations (standalone, word boundary)
    r"\bWBC\b", r"\bRBC\b", r"\bHgb\b", r"\bHb\b", r"\bHct\b",
    r"\bMCV\b", r"\bMCH\b", r"\bMCHC\b", r"\bRDW\b", r"\bMPV\b",
    r"\bAST\b", r"\bALT\b", r"\bALP\b", r"\bGGT\b", r"\bBUN\b",
    r"\beGFR\b", r"\bCRP\b", r"\bESR\b", r"\bTSH\b", r"\bINR\b",
    r"\bPTT\b", r"\bPT\b", r"\bPSA\b", r"\bBNP\b",
]

# Value + unit pattern
# Matches: 15.2 × 10³/µL, 10.8 g/dL, 245 × 10³/µL, 3.5 mmol/L, etc.
VALUE_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*"
    r"(×\s*10[³²⁴⁵⁶]?/?[µu]?L|"
    r"g/dL|mg/dL|mmol/L|mEq/L|IU/L|U/L|ng/mL|pg/mL|"
    r"µmol/L|umol/L|mL/min|mm/hr|%|cells/µL|cells/uL|"
    r"× 10³/µL|× 10³/uL|x10\^3/uL|×10\^3/µL|"
    r"× 10⁹/L|× 10¹²/L|fL|pg|g/L|µg/L|ug/L"
    r")",
    re.IGNORECASE,
)

# Range pattern: 4.5–11.0, 13.5-17.5, 150–400
RANGE_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*[–\-−to]+\s*(\d+\.?\d*)"
)


@dataclass
class BiomarkerReference:
    """A biomarker reference identified in AI-generated text."""
    biomarker_name: str
    stated_value: Optional[float] = None
    stated_unit: Optional[str] = None
    stated_range: Optional[tuple] = None  # (low, high)
    span: tuple = (0, 0)  # (start_char, end_char)


@dataclass
class AnnotatedText:
    """Text annotated with biomarker references."""
    original_text: str
    biomarkers: list[BiomarkerReference] = field(default_factory=list)


class LabExplanationHandler:
    """
    Identifies biomarker references in AI-generated lab result explanations.
    Improves claim extraction by tagging biomarker-value pairs.
    """

    def __init__(self):
        # Compile biomarker name pattern
        combined = "|".join(f"(?:{p})" for p in BIOMARKER_PATTERNS)
        self._biomarker_re = re.compile(combined, re.IGNORECASE)

    def preprocess(self, text: str) -> AnnotatedText:
        """
        Identify biomarker references in AI-generated lab explanation text.

        Args:
            text: AI-generated lab result explanation text.

        Returns:
            AnnotatedText with tagged biomarker references.
        """
        biomarkers = []

        for match in self._biomarker_re.finditer(text):
            biomarker_name = match.group().strip()
            span_start = match.start()
            span_end = match.end()

            # Look for value+unit in the surrounding context (within ~80 chars after the name)
            context_after = text[span_end:span_end + 100]
            stated_value = None
            stated_unit = None
            stated_range = None

            # Find value + unit
            val_match = VALUE_PATTERN.search(context_after)
            if val_match:
                try:
                    stated_value = float(val_match.group(1))
                    stated_unit = val_match.group(2).strip()
                except (ValueError, IndexError):
                    pass

            # Find reference range — look in wider context around the biomarker
            context_wide = text[max(0, span_start - 20):span_end + 200]
            range_matches = list(RANGE_PATTERN.finditer(context_wide))
            if range_matches:
                # Take the first range found (most likely the reference range)
                rm = range_matches[0]
                try:
                    low = float(rm.group(1))
                    high = float(rm.group(2))
                    if high > low:  # sanity check
                        stated_range = (low, high)
                except (ValueError, IndexError):
                    pass

            ref = BiomarkerReference(
                biomarker_name=biomarker_name,
                stated_value=stated_value,
                stated_unit=stated_unit,
                stated_range=stated_range,
                span=(span_start, span_end),
            )
            biomarkers.append(ref)

        # Deduplicate by span overlap
        biomarkers = self._deduplicate_overlapping(biomarkers)

        return AnnotatedText(
            original_text=text,
            biomarkers=biomarkers,
        )

    def _deduplicate_overlapping(self, refs: list[BiomarkerReference]) -> list[BiomarkerReference]:
        """Remove overlapping biomarker references, keeping the longer match."""
        if not refs:
            return refs

        # Sort by span start, then by span length (longest first)
        refs.sort(key=lambda r: (r.span[0], -(r.span[1] - r.span[0])))

        result = [refs[0]]
        for ref in refs[1:]:
            prev = result[-1]
            # If this ref overlaps with previous, skip it (previous is longer)
            if ref.span[0] < prev.span[1]:
                continue
            result.append(ref)

        return result
