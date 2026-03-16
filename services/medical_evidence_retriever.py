"""
Medical Evidence Retriever for Ancestor Health.
Orchestrates evidence retrieval from PubMed and WHO in parallel.
Returns best evidence candidate per claim for the verification pipeline.

Timeout: 5 seconds per source. Both sources timeout → claim fails closed.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from .pubmed_connector import PubMedConnector, PubMedError
from .who_connector import WHOConnector
from .health_claim_extractor import ExtractedClaim

logger = logging.getLogger(__name__)

EVIDENCE_TIMEOUT = 5.0  # seconds per source


@dataclass
class EvidenceCandidate:
    """A piece of evidence retrieved from a medical source."""
    source_url: str
    source_type: str  # "pubmed_article" | "who_indicator" | "who_guideline"
    title: str
    abstract: Optional[str] = None
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    relevance_score: float = 0.0
    content_snippet: str = ""


@dataclass
class EvidenceRetrievalResult:
    """Result of evidence retrieval for a single claim."""
    claim_id: str
    candidates: list[EvidenceCandidate] = field(default_factory=list)
    best_candidate: Optional[EvidenceCandidate] = None
    sources_queried: list[str] = field(default_factory=list)
    total_results: int = 0
    retrieval_latency_ms: int = 0


class MedicalEvidenceRetriever:
    """
    Orchestrates evidence retrieval from PubMed and WHO.
    Queries sources in parallel, merges and ranks results.
    """

    def __init__(self):
        self.pubmed = PubMedConnector()
        self.who = WHOConnector()

    async def retrieve(
        self,
        claim: ExtractedClaim,
        sources: list[str] = None,
        max_results_per_source: int = 5,
    ) -> EvidenceRetrievalResult:
        """
        Retrieve evidence for a single claim from configured sources.

        Args:
            claim: The extracted claim to find evidence for.
            sources: List of sources to query ("pubmed", "who"). Defaults to both.
            max_results_per_source: Max results per source.

        Returns:
            EvidenceRetrievalResult with ranked candidates.
        """
        if sources is None:
            sources = ["pubmed", "who"]

        start_time = time.monotonic()
        sources_queried = []
        all_candidates = []

        # Build async tasks for each source
        tasks = {}
        if "pubmed" in sources:
            tasks["pubmed"] = asyncio.create_task(
                self._query_pubmed(claim.claim_text)
            )
            sources_queried.append("pubmed")
        if "who" in sources:
            tasks["who"] = asyncio.create_task(
                self._query_who(claim.claim_text)
            )
            sources_queried.append("who")

        # Gather results with timeout
        if tasks:
            results = await asyncio.gather(
                *tasks.values(), return_exceptions=True
            )

            for source_name, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(f"{source_name} query failed for claim '{claim.claim_text[:50]}': {result}")
                    continue
                if result:
                    all_candidates.extend(result[:max_results_per_source])

        # Rank all candidates by relevance
        all_candidates.sort(key=lambda c: c.relevance_score, reverse=True)

        best = all_candidates[0] if all_candidates else None
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return EvidenceRetrievalResult(
            claim_id=claim.claim_id,
            candidates=all_candidates,
            best_candidate=best,
            sources_queried=sources_queried,
            total_results=len(all_candidates),
            retrieval_latency_ms=elapsed_ms,
        )

    async def _query_pubmed(self, claim_text: str) -> list[EvidenceCandidate]:
        """Query PubMed and convert results to EvidenceCandidates."""
        try:
            result = await asyncio.wait_for(
                self.pubmed.search(claim_text),
                timeout=EVIDENCE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(f"PubMed timeout for: {claim_text[:50]}")
            return []

        candidates = []
        for article in result.articles[:5]:
            candidates.append(
                EvidenceCandidate(
                    source_url=article.source_url,
                    source_type="pubmed_article",
                    title=article.title,
                    abstract=article.abstract,
                    publication_date=article.publication_date,
                    doi=article.doi,
                    relevance_score=self.pubmed._compute_relevance(article, claim_text),
                    content_snippet=(
                        result.content_snippet
                        if article == result.best_article
                        else (article.abstract[:300] if article.abstract else article.title)
                    ),
                )
            )

        return candidates

    async def _query_who(self, claim_text: str) -> list[EvidenceCandidate]:
        """Query WHO GHO and convert results to EvidenceCandidates."""
        try:
            result = await asyncio.wait_for(
                self.who.search(claim_text),
                timeout=EVIDENCE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(f"WHO timeout for: {claim_text[:50]}")
            return []

        if not result or not result.best_match:
            return []

        candidates = [
            EvidenceCandidate(
                source_url=result.source_url,
                source_type="who_indicator",
                title=f"WHO GHO: {result.best_match.indicator_name}",
                abstract=None,
                publication_date=str(result.best_match.year) if result.best_match.year else None,
                doi=None,
                relevance_score=0.6,  # WHO data gets moderate baseline relevance
                content_snippet=result.content_snippet,
            )
        ]

        return candidates

    async def close(self):
        await self.pubmed.close()
        await self.who.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
