"""
WHO GHO OData Connector for Ancestor Health.
Queries World Health Organization Global Health Observatory for health statistics.

Built behind MedicalSourceConnector interface for API swappability.
No authentication required. Secondary priority — system works with PubMed alone.
"""

import os
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

WHO_GHO_BASE_URL = os.environ.get(
    "WHO_GHO_BASE_URL", "https://ghoapi.azureedge.net/api"
)
WHO_TIMEOUT = 5.0  # seconds


# --- Interface ---

class MedicalSourceConnector(ABC):
    """Abstract interface for medical source connectors. Swappable."""

    @abstractmethod
    async def search(self, claim_text: str) -> Optional["WHOSearchResult"]:
        """Search for evidence matching a claim."""
        ...

    @abstractmethod
    async def close(self):
        """Cleanup resources."""
        ...


# --- WHO indicator lookup table ---

INDICATOR_MAPPINGS = {
    # HIV/AIDS
    "hiv": "HIV_0000000001",
    "hiv prevalence": "HIV_0000000001",
    "hiv incidence": "HIV_0000000026",
    # Malaria
    "malaria": "MALARIA_EST_INCIDENCE",
    "malaria incidence": "MALARIA_EST_INCIDENCE",
    "malaria mortality": "MALARIA_EST_DEATHS",
    # Tuberculosis
    "tuberculosis": "MDG_0000000020",
    "tb mortality": "MDG_0000000020",
    "tb incidence": "TB_e_inc_num",
    # Maternal health
    "maternal mortality": "MDG_0000000026",
    "maternal mortality ratio": "MDG_0000000026",
    # Child health
    "neonatal mortality": "MDG_0000000007",
    "infant mortality": "MDG_0000000001",
    "under-five mortality": "MDG_0000000007",
    # Life expectancy
    "life expectancy": "WHOSIS_000001",
    "healthy life expectancy": "WHOSIS_000002",
    # Non-communicable diseases
    "diabetes prevalence": "NCD_BMI_30A",
    "obesity": "NCD_BMI_30A",
    "blood pressure": "NCD_HYP_PREVALENCE_A",
    "hypertension": "NCD_HYP_PREVALENCE_A",
    # Immunization
    "vaccination": "WHS4_100",
    "immunization": "WHS4_100",
    # Water and sanitation
    "clean water": "WSH_SANITATION_SAFELY_MANAGED",
    "sanitation": "WSH_SANITATION_SAFELY_MANAGED",
}


@dataclass
class WHODataPoint:
    """A data point from WHO GHO."""
    indicator_code: str
    indicator_name: str
    country: Optional[str]
    year: Optional[int]
    value: Optional[float]
    display_value: str
    source_url: str


@dataclass
class WHOSearchResult:
    """Result from WHO GHO search."""
    claim_text: str
    indicator_code: Optional[str]
    data_points: list[WHODataPoint]
    best_match: Optional[WHODataPoint]
    content_snippet: str
    source_url: str
    source_type: str = "who_indicator"


class WHOConnector(MedicalSourceConnector):
    """
    WHO GHO OData client for health statistics.
    Maps claims to indicator codes via lookup table.
    """

    def __init__(self):
        self.base_url = WHO_GHO_BASE_URL.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=WHO_TIMEOUT)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def search(self, claim_text: str) -> Optional[WHOSearchResult]:
        """
        Search WHO GHO for data matching a health claim.

        Returns WHOSearchResult if relevant indicator found, None otherwise.
        """
        # Map claim to indicator code
        indicator_code = self._map_claim_to_indicator(claim_text)
        if not indicator_code:
            return None

        try:
            data_points = await self._query_indicator(indicator_code)
            if not data_points:
                return None

            # Select best data point (most recent)
            best = max(data_points, key=lambda dp: dp.year or 0)

            # Build content snippet
            snippet = self._build_snippet(best, claim_text)
            source_url = f"https://www.who.int/data/gho/data/indicators/indicator-details/GHO/{indicator_code}"

            return WHOSearchResult(
                claim_text=claim_text,
                indicator_code=indicator_code,
                data_points=data_points,
                best_match=best,
                content_snippet=snippet,
                source_url=source_url,
            )

        except httpx.TimeoutException:
            logger.warning(f"WHO GHO query timed out for indicator: {indicator_code}")
            return None
        except Exception as e:
            logger.warning(f"WHO GHO query failed: {e}")
            return None

    def _map_claim_to_indicator(self, claim_text: str) -> Optional[str]:
        """Map claim text to a WHO GHO indicator code."""
        claim_lower = claim_text.lower()

        # Direct match on indicator keywords
        best_match = None
        best_match_len = 0
        for keyword, code in INDICATOR_MAPPINGS.items():
            if keyword in claim_lower and len(keyword) > best_match_len:
                best_match = code
                best_match_len = len(keyword)

        return best_match

    async def _query_indicator(
        self, indicator_code: str, country: Optional[str] = None
    ) -> list[WHODataPoint]:
        """Query WHO GHO for a specific indicator."""
        client = await self._get_client()

        url = f"{self.base_url}/{indicator_code}"
        params = {"$top": "10", "$orderby": "TimeDim desc"}

        if country:
            params["$filter"] = f"SpatialDim eq '{country}'"

        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        data_points = []
        for item in data.get("value", []):
            try:
                dp = WHODataPoint(
                    indicator_code=indicator_code,
                    indicator_name=item.get("IndicatorCode", indicator_code),
                    country=item.get("SpatialDim"),
                    year=int(item["TimeDim"]) if item.get("TimeDim") else None,
                    value=float(item["NumericValue"]) if item.get("NumericValue") else None,
                    display_value=str(item.get("Value", "")),
                    source_url=f"https://www.who.int/data/gho/data/indicators/indicator-details/GHO/{indicator_code}",
                )
                data_points.append(dp)
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping WHO data point: {e}")
                continue

        return data_points

    def _build_snippet(self, data_point: WHODataPoint, claim_text: str) -> str:
        """Build a content snippet from a WHO data point."""
        parts = []

        if data_point.indicator_name:
            parts.append(f"WHO Global Health Observatory: {data_point.indicator_name}")

        if data_point.country:
            parts.append(f"Country: {data_point.country}")

        if data_point.year:
            parts.append(f"Year: {data_point.year}")

        if data_point.display_value:
            parts.append(f"Value: {data_point.display_value}")

        return ". ".join(parts) + "."

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
