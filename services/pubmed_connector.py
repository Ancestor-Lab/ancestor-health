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
QUERY_TIMEOUT = 5.0  # seconds
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
        Strips stop words, extracts medical terms, adds publication type filters.
        """
        # Lowercase and extract words
        words = re.findall(r'[a-zA-Z0-9]+', claim_text.lower())

        # Remove stop words
        medical_terms = [w for w in words if w not in STOP_WORDS and len(w) > 2]

        if not medical_terms:
            # Fallback: use all non-trivial words
            medical_terms = [w for w in words if len(w) > 2]

        if not medical_terms:
            return claim_text

        # Build query: combine terms with AND, add review/guideline filter
        term_query = " AND ".join(medical_terms[:6])  # Limit to 6 key terms
        query = f"({term_query}) AND (review[pt] OR guideline[pt] OR systematic review[pt] OR meta-analysis[pt])"

        return query

    def _build_fallback_query(self, claim_text: str) -> str:
        """Build a broader fallback query without publication type filters."""
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

        # Abstract
        abstract_parts = []
        for abs_text in elem.findall(".//AbstractText"):
            label = abs_text.get("Label", "")
            text = abs_text.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

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
        """Extract the most relevant paragraph from the abstract."""
        if not article.abstract:
            return article.title

        # Split abstract into paragraphs/sections
        paragraphs = [p.strip() for p in article.abstract.split(". ") if len(p.strip()) > 20]

        if not paragraphs:
            return article.abstract[:500]

        # Score each paragraph by term overlap with the claim
        claim_terms = set(
            w.lower() for w in re.findall(r'[a-zA-Z0-9]+', claim_text)
            if w.lower() not in STOP_WORDS and len(w) > 2
        )

        best_para = article.abstract[:500]
        best_score = 0.0

        for para in paragraphs:
            para_terms = set(
                w.lower() for w in re.findall(r'[a-zA-Z0-9]+', para)
                if w not in STOP_WORDS and len(w) > 2
            )
            if para_terms:
                overlap = len(claim_terms & para_terms) / max(len(claim_terms), 1)
                if overlap > best_score:
                    best_score = overlap
                    best_para = para

        # Return the best paragraph, ensuring it ends with a period
        if not best_para.endswith("."):
            best_para += "."
        return best_para

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
