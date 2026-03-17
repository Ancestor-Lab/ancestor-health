"""
PubMed E-utilities Connector for Ancestor Health.
Queries NCBI PubMed for peer-reviewed medical evidence.

Uses ESearch + EFetch pipeline with medical term extraction.
Rate limits: 3 req/s without API key, 10 req/s with key.
Timeout: 5 seconds per query. Fails closed on error.
"""

import os
import re
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# NCBI E-utilities base URL
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Environment config
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")
NCBI_TOOL_NAME = os.environ.get("NCBI_TOOL_NAME", "ancestor-health")

# Request config
QUERY_TIMEOUT = 15.0  # seconds (increased for VPC egress latency)
MAX_RESULTS = 10

# Stop words to strip from claim text before building queries
STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "but", "and", "or", "if", "because", "until", "while", "that",
    "this", "these", "those", "it", "its", "which", "who", "whom",
    "what", "about", "also", "your", "their", "his", "her", "my",
})

# Medical term mapping: common phrases/biomarkers → standard PubMed search terms
MEDICAL_TERM_MAP = {
    # Biomarker names → medical terms
    "white blood cell": "leukocyte",
    "white blood cells": "leukocyte",
    "wbc": "leukocyte",
    "red blood cell": "erythrocyte",
    "red blood cells": "erythrocyte",
    "rbc": "erythrocyte",
    "hemoglobin": "hemoglobin",
    "hgb": "hemoglobin",
    "hematocrit": "hematocrit",
    "hct": "hematocrit",
    "platelet": "platelet",
    "platelets": "platelet count",
    "plt": "platelet",
    "blood sugar": "blood glucose",
    "blood glucose": "blood glucose",
    "fasting glucose": "fasting blood glucose",
    "a1c": "glycated hemoglobin",
    "hba1c": "glycated hemoglobin",
    "hemoglobin a1c": "glycated hemoglobin",
    "cholesterol": "cholesterol",
    "ldl": "LDL cholesterol",
    "hdl": "HDL cholesterol",
    "triglycerides": "triglycerides",
    "creatinine": "creatinine",
    "bun": "blood urea nitrogen",
    "gfr": "glomerular filtration rate",
    "egfr": "estimated glomerular filtration rate",
    "alt": "alanine aminotransferase",
    "ast": "aspartate aminotransferase",
    "bilirubin": "bilirubin",
    "albumin": "serum albumin",
    "tsh": "thyroid stimulating hormone",
    "t3": "triiodothyronine",
    "t4": "thyroxine",
    "psa": "prostate specific antigen",
    "crp": "C-reactive protein",
    "esr": "erythrocyte sedimentation rate",
    "inr": "international normalized ratio",
    "ptt": "partial thromboplastin time",
    "d-dimer": "D-dimer",
    "ferritin": "ferritin",
    "iron": "serum iron",
    "vitamin d": "vitamin D",
    "vitamin b12": "vitamin B12",
    "folate": "folate",
    "potassium": "potassium",
    "sodium": "sodium",
    "calcium": "calcium",
    "magnesium": "magnesium",
    "blood pressure": "blood pressure",
    "systolic": "systolic blood pressure",
    "diastolic": "diastolic blood pressure",
    "heart rate": "heart rate",
    "bmi": "body mass index",
    "body mass index": "body mass index",
    # Clinical conditions from lab context
    "elevated white blood cell": "leukocytosis",
    "high white blood cell": "leukocytosis",
    "low white blood cell": "leukopenia",
    "elevated wbc": "leukocytosis",
    "low wbc": "leukopenia",
    "low hemoglobin": "anemia",
    "low hgb": "anemia",
    "below normal hemoglobin": "anemia",
    "low platelet": "thrombocytopenia",
    "low platelets": "thrombocytopenia",
    "high platelet": "thrombocytosis",
    "high platelets": "thrombocytosis",
    "high blood sugar": "hyperglycemia",
    "low blood sugar": "hypoglycemia",
    "high cholesterol": "hypercholesterolemia",
    "high blood pressure": "hypertension",
    "low blood pressure": "hypotension",
    "high creatinine": "renal impairment",
    "low gfr": "chronic kidney disease",
    "high tsh": "hypothyroidism",
    "low tsh": "hyperthyroidism",
    "high psa": "prostate cancer screening",
    "high crp": "inflammation",
    "high esr": "inflammation",
    # Common clinical phrases → keep simple PubMed-friendly terms
    "bacterial infection": "bacterial infection",
    "viral infection": "viral infection",
    "active infection": "infection",
    "normal range": "reference values",
    "normal reference range": "reference values",
    "below normal": "anemia",  # context-dependent but common
    "above normal": "elevated",
    "slightly below": None,  # noise — skip
    "slightly above": None,  # noise — skip
    "adult males": None,  # noise — skip
    "adult females": None,  # noise — skip
    "complete blood count": "complete blood count",
    "cbc": "complete blood count",
    "metabolic panel": "metabolic panel",
    "liver function": "liver function",
    "kidney function": "renal function",
    "thyroid function": "thyroid function",
    "lipid panel": "lipid panel",
    # Anemia-specific
    "mild anemia": "anemia",
    "indicating mild anemia": "anemia",
    "indicating anemia": "anemia",
    # Common health conditions and associations
    "cardiovascular disease": "cardiovascular disease",
    "heart disease": "cardiovascular disease",
    "coronary artery disease": "coronary artery disease",
    "heart attack": "myocardial infarction",
    "myocardial infarction": "myocardial infarction",
    "stroke": "stroke",
    "lung cancer": "lung cancer",
    "breast cancer": "breast cancer",
    "colon cancer": "colorectal cancer",
    "colorectal cancer": "colorectal cancer",
    "prostate cancer": "prostate cancer",
    "diabetes": "diabetes mellitus",
    "type 2 diabetes": "type 2 diabetes mellitus",
    "type 1 diabetes": "type 1 diabetes mellitus",
    "asthma": "asthma",
    "copd": "chronic obstructive pulmonary disease",
    "chronic obstructive pulmonary disease": "COPD",
    "pneumonia": "pneumonia",
    "sepsis": "sepsis",
    "kidney disease": "chronic kidney disease",
    "liver disease": "liver disease",
    "alzheimer": "Alzheimer disease",
    "dementia": "dementia",
    "depression": "depression",
    "anxiety": "anxiety",
    "obesity": "obesity",
    "osteoporosis": "osteoporosis",
    "arthritis": "arthritis",
    "rheumatoid arthritis": "rheumatoid arthritis",
    # Risk factors and interventions
    "physical exercise": "exercise",
    "physical activity": "physical activity",
    "smoking": "smoking",
    "tobacco": "tobacco",
    "alcohol": "alcohol",
    "blood thinner": "anticoagulant",
    "aspirin": "aspirin",
    "statin": "statin",
    "antibiotic": "antibiotics",
    "vaccine": "vaccination",
    "reduces the risk": None,  # noise
    "increase the risk": None,  # noise
    "associated with": None,  # noise
}

# Phrases to strip from claims before query building (noise)
NOISE_PHRASES = [
    r'\b\d+\.?\d*\s*(?:×\s*10[³⁹]\s*/?µ?[Ll]|g/d[Ll]|mg/d[Ll]|mmol/[Ll]|mEq/[Ll]|U/[Ll]|IU/[Ll]|ng/m[Ll]|pg/m[Ll]|%)\b',
    r'\b\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*\b',  # ranges like "4.5-11.0"
    r'\b(?:range|level|count|value|result|test|shows?|indicates?|suggests?|may|could|would|within|outside|reading|falls?|possible|slightly|elevated|normal|above|below)\b',
]

# Publication type weights for ranking
PUB_TYPE_WEIGHTS = {
    "Systematic Review": 1.0,
    "Meta-Analysis": 1.0,
    "Practice Guideline": 0.95,
    "Guideline": 0.9,
    "Review": 0.8,
    "Randomized Controlled Trial": 0.7,
    "Clinical Trial": 0.65,
    "Observational Study": 0.5,
    "Cohort Study": 0.5,
    "Case-Control Study": 0.45,
    "Case Reports": 0.3,
}


class PubMedError(Exception):
    """Raised when PubMed query fails. Maps to VERIFICATION_FAILURE."""
    pass


class PubMedNoResultsError(PubMedError):
    """Raised when PubMed returns zero results."""
    pass


class PubMedTimeoutError(PubMedError):
    """Raised when PubMed query times out."""
    pass


@dataclass
class PubMedArticle:
    """Parsed PubMed article from EFetch XML."""
    pmid: str
    title: str
    abstract: str
    conclusion_snippet: str = ""  # CONCLUSIONS/RESULTS section for NLI input
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    pub_types: list[str] = field(default_factory=list)
    source_url: str = ""

    def __post_init__(self):
        self.source_url = f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


@dataclass
class PubMedSearchResult:
    """Result from PubMed search with ranked articles."""
    query: str
    articles: list[PubMedArticle]
    best_article: Optional[PubMedArticle]
    content_snippet: str
    relevance_score: float
    total_found: int


class PubMedConnector:
    """
    PubMed E-utilities client for medical evidence retrieval.
    ESearch → EFetch pipeline with relevance ranking.
    """

    def __init__(self):
        self.api_key = NCBI_API_KEY
        self.email = NCBI_EMAIL
        self.tool_name = NCBI_TOOL_NAME
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=QUERY_TIMEOUT)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _build_common_params(self) -> dict:
        """Build common NCBI E-utilities parameters."""
        params = {"tool": self.tool_name}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def build_search_query(self, claim_text: str) -> str:
        """
        Build a PubMed search query from claim text.
        Maps common biomarker/clinical phrases to standard medical terms,
        strips numeric values/units/noise, then builds a focused query.
        """
        claim_lower = claim_text.lower()

        # Step 1: Extract mapped medical terms (longest match first)
        mapped_terms = []
        sorted_keys = sorted(MEDICAL_TERM_MAP.keys(), key=len, reverse=True)
        used_spans = []

        for phrase in sorted_keys:
            idx = claim_lower.find(phrase)
            if idx >= 0:
                end = idx + len(phrase)
                # Check no overlap with already-matched spans
                if not any(s <= idx < e or s < end <= e for s, e in used_spans):
                    mapped_term = MEDICAL_TERM_MAP[phrase]
                    if mapped_term is not None and mapped_term not in mapped_terms:
                        mapped_terms.append(mapped_term)
                    used_spans.append((idx, end))

        # Step 2: Strip noise (numbers, units, ranges, filler verbs)
        cleaned = claim_lower
        for pattern in NOISE_PHRASES:
            cleaned = re.sub(pattern, ' ', cleaned)

        # Step 3: Extract remaining medical words (not stop words, not already mapped)
        # Prioritize longer words (more likely to be medical terms)
        remaining_words = re.findall(r'[a-zA-Z]+', cleaned)
        mapped_lower = {t.lower() for t in mapped_terms}
        extra_terms_set = set()
        extra_terms_list = []
        for w in remaining_words:
            wl = w.lower()
            if (wl not in STOP_WORDS
                    and len(wl) > 3
                    and wl not in mapped_lower
                    and wl not in extra_terms_set
                    and not any(wl in t.lower() for t in mapped_terms)):
                extra_terms_set.add(wl)
                extra_terms_list.append(wl)
        # Sort by length descending — longer words are more specific/medical
        extra_terms = sorted(extra_terms_list, key=len, reverse=True)

        # Step 4: Combine — mapped terms take priority, limit to 3 AND terms total
        query_terms = mapped_terms[:3]
        slots_left = 3 - len(query_terms)
        if slots_left > 0:
            query_terms.extend(extra_terms[:slots_left])

        if not query_terms:
            # Ultimate fallback: raw words
            words = re.findall(r'[a-zA-Z0-9]+', claim_lower)
            query_terms = [w for w in words if w not in STOP_WORDS and len(w) > 2][:5]

        if not query_terms:
            return claim_text

        # Step 5: For reference range claims, add "reference values" to find
        # articles that define normal ranges rather than clinical management
        is_reference_claim = any(
            phrase in claim_lower
            for phrase in ["normal range", "reference range", "within the normal",
                           "below the normal", "above the normal", "falls within"]
        )
        if is_reference_claim and "reference values" not in query_terms:
            # Append as OR to broaden, not narrow
            ref_boost = ' AND ("reference values"[MeSH] OR "reference interval")'
        else:
            ref_boost = ""

        # Step 6: Build PubMed query with publication type filter
        term_query = " AND ".join(f'"{t}"' if ' ' in t else t for t in query_terms)
        query = f"({term_query}{ref_boost}) AND (review[pt] OR guideline[pt] OR systematic review[pt] OR meta-analysis[pt])"

        return query

    def _build_fallback_query(self, claim_text: str) -> str:
        """Build a broader fallback query without publication type filters.
        Uses the same medical term mapping but drops the pub type filter."""
        # Reuse the mapped query but strip the publication type filter
        full_query = self.build_search_query(claim_text)
        # Remove the pub type filter suffix
        if ") AND (review[pt]" in full_query:
            return full_query.split(") AND (review[pt]")[0] + ")"
        # Final fallback: raw words
        words = re.findall(r'[a-zA-Z0-9]+', claim_text.lower())
        medical_terms = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        if not medical_terms:
            medical_terms = [w for w in words if len(w) > 2]
        return " AND ".join(medical_terms[:5]) if medical_terms else claim_text

    async def search(self, claim_text: str) -> PubMedSearchResult:
        """
        Search PubMed for evidence matching a claim.

        Args:
            claim_text: The health claim to find evidence for.

        Returns:
            PubMedSearchResult with ranked articles.

        Raises:
            PubMedNoResultsError: No results found.
            PubMedTimeoutError: Query timed out.
            PubMedError: Other PubMed errors.
        """
        query = self.build_search_query(claim_text)

        try:
            # Step 1: ESearch to get PMIDs
            pmids = await self._esearch(query)

            # If no results with strict query, try fallback without pub type filter
            if not pmids:
                fallback_query = self._build_fallback_query(claim_text)
                pmids = await self._esearch(fallback_query)
                if pmids:
                    query = fallback_query

            if not pmids:
                raise PubMedNoResultsError(
                    f"No PubMed results for claim: {claim_text[:100]}"
                )

            # Step 2: EFetch to get article details
            articles = await self._efetch(pmids)

            if not articles:
                raise PubMedNoResultsError(
                    f"EFetch returned no parseable articles for PMIDs: {pmids}"
                )

            # Step 3: Rank by relevance to claim
            ranked = self._rank_articles(articles, claim_text)

            best = ranked[0]
            snippet = self._extract_best_snippet(best, claim_text)
            relevance = self._compute_relevance(best, claim_text)

            return PubMedSearchResult(
                query=query,
                articles=ranked,
                best_article=best,
                content_snippet=snippet,
                relevance_score=relevance,
                total_found=len(ranked),
            )

        except httpx.TimeoutException as e:
            raise PubMedTimeoutError(f"PubMed query timed out: {e}") from e
        except (PubMedNoResultsError, PubMedTimeoutError):
            raise
        except Exception as e:
            raise PubMedError(f"PubMed query failed: {e}") from e

    async def _esearch(self, query: str) -> list[str]:
        """Run ESearch and return list of PMIDs."""
        client = await self._get_client()
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(MAX_RESULTS),
            "sort": "relevance",
            "retmode": "json",
            **self._build_common_params(),
        }

        resp = await client.get(f"{EUTILS_BASE}esearch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()

        id_list = data.get("esearchresult", {}).get("idlist", [])
        return id_list

    async def _efetch(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch article details for a list of PMIDs."""
        client = await self._get_client()
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            **self._build_common_params(),
        }

        resp = await client.get(f"{EUTILS_BASE}efetch.fcgi", params=params)
        resp.raise_for_status()

        return self._parse_efetch_xml(resp.text)

    def _parse_efetch_xml(self, xml_text: str) -> list[PubMedArticle]:
        """Parse EFetch XML response into PubMedArticle objects."""
        articles = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed XML: {e}")
            return articles

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                articles.append(self._parse_single_article(article_elem))
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")
                continue

        return articles

    def _parse_single_article(self, elem: ET.Element) -> PubMedArticle:
        """Parse a single PubmedArticle XML element."""
        # PMID
        pmid_elem = elem.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else "unknown"

        # Title
        title_elem = elem.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else ""

        # Abstract — parse structured sections for conclusion extraction
        abstract_parts = []
        section_map = {}  # label → text for structured abstracts
        for abs_text in elem.findall(".//AbstractText"):
            label = abs_text.get("Label", "")
            text = abs_text.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
                section_map[label.upper()] = text
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Extract conclusion snippet: CONCLUSIONS > RESULTS > FINDINGS > last section
        conclusion_snippet = ""
        for priority_label in ["CONCLUSIONS", "CONCLUSION", "RESULTS", "FINDINGS", "RESULTS AND CONCLUSIONS"]:
            if priority_label in section_map and section_map[priority_label].strip():
                conclusion_snippet = section_map[priority_label].strip()
                break
        if not conclusion_snippet and section_map:
            # Use the last labelled section (conclusions typically come last)
            last_label = list(section_map.keys())[-1]
            conclusion_snippet = section_map[last_label].strip()
        if not conclusion_snippet and abstract:
            # Unstructured abstract: use the last paragraph (after last period-space)
            sentences = abstract.rsplit(". ", 1)
            conclusion_snippet = sentences[-1].strip() if len(sentences) > 1 else abstract[-500:]

        # Publication date
        pub_date = self._extract_pub_date(elem)

        # DOI
        doi = None
        for id_elem in elem.findall(".//ArticleId"):
            if id_elem.get("IdType") == "doi":
                doi = id_elem.text
                break

        # Journal
        journal_elem = elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else None

        # Publication types
        pub_types = []
        for pt in elem.findall(".//PublicationType"):
            if pt.text:
                pub_types.append(pt.text)

        return PubMedArticle(
            pmid=pmid,
            title=title or "",
            abstract=abstract,
            conclusion_snippet=conclusion_snippet,
            publication_date=pub_date,
            doi=doi,
            journal=journal,
            pub_types=pub_types,
        )

    def _extract_pub_date(self, elem: ET.Element) -> Optional[str]:
        """Extract publication date from article XML."""
        # Try PubDate first
        for date_path in [".//PubDate", ".//ArticleDate"]:
            date_elem = elem.find(date_path)
            if date_elem is not None:
                year = date_elem.findtext("Year")
                month = date_elem.findtext("Month", "01")
                day = date_elem.findtext("Day", "01")
                if year:
                    # Convert month name to number if needed
                    try:
                        month_num = int(month)
                    except ValueError:
                        try:
                            month_num = datetime.strptime(month[:3], "%b").month
                        except ValueError:
                            month_num = 1
                    try:
                        day_num = int(day)
                    except ValueError:
                        day_num = 1
                    return f"{year}-{month_num:02d}-{day_num:02d}"
        return None

    def _rank_articles(self, articles: list[PubMedArticle], claim_text: str) -> list[PubMedArticle]:
        """Rank articles by relevance to the claim."""
        scored = []
        for article in articles:
            score = self._compute_relevance(article, claim_text)
            scored.append((score, article))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [article for _, article in scored]

    def _compute_relevance(self, article: PubMedArticle, claim_text: str) -> float:
        """
        Compute relevance score (0.0-1.0) for an article against a claim.
        Uses Jaccard similarity on medical terms + publication type weight + recency.
        """
        # Term overlap (Jaccard on non-stop words)
        claim_terms = set(
            w.lower() for w in re.findall(r'[a-zA-Z0-9]+', claim_text)
            if w.lower() not in STOP_WORDS and len(w) > 2
        )
        article_text = f"{article.title} {article.abstract}".lower()
        article_terms = set(
            w for w in re.findall(r'[a-zA-Z0-9]+', article_text)
            if w not in STOP_WORDS and len(w) > 2
        )

        if claim_terms and article_terms:
            jaccard = len(claim_terms & article_terms) / len(claim_terms | article_terms)
        else:
            jaccard = 0.0

        # Publication type weight
        pub_type_score = 0.3  # default for unknown types
        for pt in article.pub_types:
            weight = PUB_TYPE_WEIGHTS.get(pt, 0.0)
            if weight > pub_type_score:
                pub_type_score = weight

        # Recency bonus
        recency_score = 0.5
        if article.publication_date:
            try:
                pub_year = int(article.publication_date[:4])
                current_year = datetime.now().year
                age = current_year - pub_year
                if age <= 2:
                    recency_score = 1.0
                elif age <= 5:
                    recency_score = 0.7
                else:
                    recency_score = 0.4
            except (ValueError, IndexError):
                pass

        # Weighted combination: term overlap 50%, pub type 30%, recency 20%
        relevance = (jaccard * 0.5) + (pub_type_score * 0.3) + (recency_score * 0.2)
        return round(min(relevance, 1.0), 4)

    def _extract_best_snippet(self, article: PubMedArticle, claim_text: str) -> str:
        """Extract the most relevant snippet from the abstract for NLI input.
        Prefers conclusion/results sections over full abstract — these contain
        the finding statement which scores much higher on NLI entailment."""
        # Priority 1: Use conclusion_snippet if available and non-trivial
        if article.conclusion_snippet and len(article.conclusion_snippet) > 30:
            snippet = article.conclusion_snippet[:500]
            if not snippet.endswith("."):
                snippet += "."
            return snippet

        if not article.abstract:
            return article.title

        # Priority 2: Last paragraph of unstructured abstract (conclusion position)
        paragraphs = [p.strip() for p in article.abstract.split(". ") if len(p.strip()) > 20]
        if paragraphs:
            # Take the last 1-2 sentences — conclusions come last
            last_para = paragraphs[-1]
            if not last_para.endswith("."):
                last_para += "."
            return last_para

        return article.abstract[:500]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
