"""
Phase 3: Hardened Source Fetching Module
Build the "eyes" of the system with caching, retries, and audit snapshots.

This module handles HTTP requests with:
- Deterministic caching with URL normalization
- Retry logic with exponential backoff for transient failures
- Audit-grade SHA-256 snapshotting
- Structured availability signals
- Fail-closed error handling
"""

import requests
import logging
import hashlib
import time
import re
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup, Comment

# Phase 3: Import retry capabilities
try:
    from tenacity import (
        retry, stop_after_attempt, wait_exponential,
        retry_if_exception_type, before_sleep_log
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Tenacity not available - retry functionality disabled")

# Phase 3: Import cache service
from .cache_service import CacheService

# Configure logging
logger = logging.getLogger(__name__)

# Phase 3: Exception types for retry logic
class TransientFetchError(Exception):
    """Transient error that should be retried (timeouts, 5xx, 429)."""
    pass

class PermanentFetchError(Exception):
    """Permanent error that should NOT be retried (404, 403, DNS)."""
    pass

class SourceFetcher:
    """
    Phase 3: Hardened URL fetching with caching, retries, and audit snapshots.

    Features:
    - Deterministic caching with TTL
    - Exponential backoff retry for transient failures
    - SHA-256 audit snapshots
    - Structured availability signals
    - Fail-closed error handling
    """

    def __init__(self, cache_service: Optional[CacheService] = None,
                 timeout: int = 10, max_content_length: int = 2 * 1024 * 1024):
        """
        Initialize the hardened source fetcher.

        Args:
            cache_service: Cache service instance (optional)
            timeout: HTTP request timeout in seconds (increased for stability)
            max_content_length: Maximum content length in bytes (2MB for better coverage)
        """
        self.cache_service = cache_service
        self.timeout = timeout
        self.max_content_length = max_content_length

        # Enhanced user agent for Phase 3
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; AncestorCloud/3.0; +https://ancestor.cloud/about) Phase3-Hardened',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=300',
            'DNT': '1'
        }

    def _compute_snapshot_hash(self, text: str) -> str:
        """
        Phase 3: Generate SHA-256 hash of extracted text for audit trail.

        Args:
            text: Extracted text content

        Returns:
            SHA-256 hash as hex string
        """
        if not text:
            return ""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def fetch_and_extract(self, url: str) -> Dict[str, Any]:
        """
        Phase 3: Orchestrates Cache -> Retry -> Snapshot -> Store pipeline.

        Args:
            url: Target URL to fetch

        Returns:
            Dict containing:
                - availability_signal: 1.0 (success) or 0.0 (failure)
                - extracted_text: Clean text content (if successful)
                - content_length: Length of extracted text
                - final_url: Final URL after redirects
                - fetch_time_ms: Time taken for fetch operation
                - fetch_status: Structured availability signal (FETCH_OK, FETCH_CACHED, etc.)
                - snapshot_hash: SHA-256 hash of extracted content
                - error_type: Error classification (if failed)
                - error_message: Detailed error message (if failed)
        """
        start_time = time.time()

        # Step 1: Check Cache First
        if self.cache_service:
            cached_result = self.cache_service.get(url)
            if cached_result:
                fetch_time_ms = int((time.time() - start_time) * 1000)
                return {
                    "availability_signal": cached_result['availability_signal'],
                    "extracted_text": cached_result['extracted_text'],
                    "content_length": cached_result['content_length'],
                    "final_url": cached_result.get('final_url', url),
                    "fetch_time_ms": fetch_time_ms,
                    "fetch_status": "FETCH_CACHED",
                    "snapshot_hash": cached_result.get('snapshot_hash',
                                                     self._compute_snapshot_hash(cached_result['extracted_text'])),
                    "cached_timestamp": cached_result.get('cached_timestamp'),
                    "original_fetch_timestamp": cached_result.get('fetch_timestamp'),
                    "error_type": None,
                    "error_message": None
                }

        # Step 2: Live Fetch with Retry Logic
        try:
            live_result = self._perform_live_fetch_with_retries(url)
            fetch_time_ms = int((time.time() - start_time) * 1000)

            # Step 3: Add timing and snapshot hash
            live_result["fetch_time_ms"] = fetch_time_ms
            if live_result["availability_signal"] > 0:
                live_result["snapshot_hash"] = self._compute_snapshot_hash(live_result["extracted_text"])
            else:
                live_result["snapshot_hash"] = ""

            # Step 4: Store successful results in cache
            if (self.cache_service and
                live_result["availability_signal"] > 0 and
                live_result.get("extracted_text")):
                self.cache_service.set(url, live_result)

            return live_result

        except Exception as e:
            logger.exception(f"Unexpected error in fetch pipeline for {url}")
            return self._error_response(
                url, start_time, "pipeline_error",
                f"Fetch pipeline error: {str(e)}"
            )

    def _extract_text_content(self, content: bytes, headers: Dict[str, str]) -> str:
        """
        Extract clean text content from HTML using BeautifulSoup.
        Removes boilerplate content like navigation, footer, scripts.

        Args:
            content: Raw HTML content bytes
            headers: HTTP response headers

        Returns:
            Clean extracted text
        """
        try:
            # Detect encoding
            encoding = 'utf-8'
            content_type = headers.get('content-type', '').lower()
            if 'charset=' in content_type:
                encoding = content_type.split('charset=')[1].split(';')[0].strip()

            # Parse HTML
            text_content = content.decode(encoding, errors='ignore')
            soup = BeautifulSoup(text_content, 'html.parser')

            # Remove unwanted elements
            unwanted_tags = [
                'script', 'style', 'nav', 'header', 'footer',
                'aside', 'menu', 'form', 'button', 'input'
            ]

            for tag in unwanted_tags:
                for element in soup.find_all(tag):
                    element.decompose()

            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            # Remove elements with common boilerplate classes/ids
            boilerplate_selectors = [
                '[class*="nav"]', '[class*="menu"]', '[class*="sidebar"]',
                '[class*="footer"]', '[class*="header"]', '[class*="advertisement"]',
                '[class*="cookie"]', '[class*="popup"]', '[id*="nav"]',
                '[id*="menu"]', '[id*="sidebar"]', '[id*="footer"]'
            ]

            for selector in boilerplate_selectors:
                for element in soup.select(selector):
                    element.decompose()

            # Extract text content
            text_content = soup.get_text(separator=' ', strip=True)

            # Clean and normalize text
            text_content = self._clean_text(text_content)

            return text_content

        except Exception as e:
            logger.warning(f"Failed to extract text content: {str(e)}")
            return ""

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Limit length for safety
        max_text_length = 50000  # 50K characters should be enough for most articles
        if len(text) > max_text_length:
            text = text[:max_text_length] + "... [truncated]"

        return text

    def _perform_live_fetch_with_retries(self, url: str) -> Dict[str, Any]:
        """
        Phase 3: Perform live HTTP fetch with intelligent retry logic.

        Retry Strategy:
        - Transient failures (timeouts, 5xx, 429): Retry with exponential backoff
        - Permanent failures (404, 403, DNS): Fail immediately, no retries
        - Max attempts: 3 (1 initial + 2 retries)
        - Backoff: 1s, 2s, 4s

        Args:
            url: Target URL to fetch

        Returns:
            Fetch result dict with availability signals
        """
        if not TENACITY_AVAILABLE:
            # Fallback to single attempt if tenacity not available
            return self._single_fetch_attempt(url)

        # Phase 3: Define retry decorator with exponential backoff
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(TransientFetchError),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )
        def _fetch_with_retries():
            return self._single_fetch_attempt(url)

        try:
            return _fetch_with_retries()
        except TransientFetchError as e:
            logger.error(f"All retry attempts exhausted for {url}: {str(e)}")
            return {
                "availability_signal": 0.0,
                "extracted_text": "",
                "content_length": 0,
                "final_url": url,
                "fetch_status": "FETCH_RETRY_EXHAUSTED",
                "error_type": "retry_exhausted",
                "error_message": f"All 3 retry attempts failed: {str(e)}"
            }
        except PermanentFetchError as e:
            logger.warning(f"Permanent fetch failure for {url}: {str(e)}")
            return {
                "availability_signal": 0.0,
                "extracted_text": "",
                "content_length": 0,
                "final_url": url,
                "fetch_status": "FETCH_PERMANENT_FAILURE",
                "error_type": "permanent_failure",
                "error_message": str(e)
            }

    def _single_fetch_attempt(self, url: str) -> Dict[str, Any]:
        """
        Phase 3: Perform single HTTP fetch attempt with error classification.

        Args:
            url: Target URL to fetch

        Returns:
            Fetch result dict

        Raises:
            TransientFetchError: For retryable failures (timeouts, 5xx, 429)
            PermanentFetchError: For non-retryable failures (404, 403, DNS)
        """
        try:
            # Validate URL format
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise PermanentFetchError(f"Invalid URL format: {url}")

            # Make HTTP request
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True,
                stream=True
            )

            final_url = response.url

            # Check content length before downloading
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_content_length:
                raise PermanentFetchError(
                    f"Content too large: {content_length} bytes > {self.max_content_length}"
                )

            # Check HTTP status codes
            if response.status_code == 200:
                # Success - download and extract content
                content = b""
                downloaded = 0

                for chunk in response.iter_content(chunk_size=8192):
                    content += chunk
                    downloaded += len(chunk)

                    # Enforce size limit during download
                    if downloaded > self.max_content_length:
                        raise PermanentFetchError(
                            f"Content exceeds size limit during download: {downloaded} > {self.max_content_length}"
                        )

                # Extract text content
                extracted_text = self._extract_text_content(content, dict(response.headers))
                content_length = len(extracted_text)

                logger.info(f"Successfully fetched {url}: {content_length} chars")

                return {
                    "availability_signal": 1.0,
                    "extracted_text": extracted_text,
                    "content_length": content_length,
                    "final_url": final_url,
                    "fetch_status": "FETCH_OK",
                    "error_type": None,
                    "error_message": None
                }

            elif response.status_code == 404:
                raise PermanentFetchError(f"Resource not found (404): {url}")

            elif response.status_code == 403:
                raise PermanentFetchError(f"Access forbidden (403): {url}")

            elif response.status_code == 401:
                raise PermanentFetchError(f"Authentication required (401): {url}")

            elif response.status_code == 429:
                # Rate limiting - transient, should retry
                raise TransientFetchError(f"Rate limited (429): {url}")

            elif 400 <= response.status_code < 500:
                # Other 4xx errors are generally permanent
                raise PermanentFetchError(f"Client error ({response.status_code}): {url}")

            elif 500 <= response.status_code < 600:
                # 5xx errors are transient, should retry
                raise TransientFetchError(f"Server error ({response.status_code}): {url}")

            else:
                # Unexpected status code
                raise TransientFetchError(f"Unexpected status ({response.status_code}): {url}")

        except requests.exceptions.Timeout:
            raise TransientFetchError(f"Request timeout: {url}")

        except requests.exceptions.ConnectionError as e:
            error_msg = str(e).lower()
            # Check for DNS resolution failures (permanent)
            if any(dns_indicator in error_msg for dns_indicator in ['name resolution failed', 'nodename nor servname provided', 'name or service not known']):
                raise PermanentFetchError(f"DNS resolution failed: {url}")
            else:
                # Other connection errors might be transient
                raise TransientFetchError(f"Connection error: {url} - {str(e)}")

        except requests.exceptions.SSLError:
            # SSL errors are generally permanent
            raise PermanentFetchError(f"SSL error: {url}")

        except requests.exceptions.TooManyRedirects:
            # Redirect loops are permanent
            raise PermanentFetchError(f"Too many redirects: {url}")

        except Exception as e:
            # Unknown errors - treat as transient to be safe
            raise TransientFetchError(f"Unknown error fetching {url}: {str(e)}")

    def _error_response(self, url: str, start_time: float, error_type: str, error_message: str) -> Dict[str, Any]:
        """
        Create standardized error response with Phase 3 structured availability signals.

        Args:
            url: Original URL
            start_time: Request start time
            error_type: Error classification
            error_message: Detailed error message

        Returns:
            Standardized error response dict with availability signals
        """
        fetch_time_ms = int((time.time() - start_time) * 1000)

        logger.warning(f"Source fetch failed for {url}: {error_type} - {error_message}")

        # Phase 3: Map error types to structured availability signals
        fetch_status_mapping = {
            "timeout": "FETCH_TIMEOUT",
            "connection_error": "FETCH_CONNECTION_ERROR",
            "dns_error": "FETCH_DNS_ERROR",
            "ssl_error": "FETCH_SSL_ERROR",
            "http_404": "FETCH_404",
            "http_403": "FETCH_403",
            "http_401": "FETCH_401",
            "http_429": "FETCH_RATE_LIMITED",
            "http_5xx": "FETCH_SERVER_ERROR",
            "content_too_large": "FETCH_CONTENT_TOO_LARGE",
            "invalid_url": "FETCH_INVALID_URL",
            "permanent_failure": "FETCH_PERMANENT_FAILURE",
            "retry_exhausted": "FETCH_RETRY_EXHAUSTED",
            "pipeline_error": "FETCH_PIPELINE_ERROR"
        }

        fetch_status = fetch_status_mapping.get(error_type, "FETCH_UNKNOWN_ERROR")

        return {
            "availability_signal": 0.0,
            "extracted_text": "",
            "content_length": 0,
            "final_url": url,
            "fetch_time_ms": fetch_time_ms,
            "fetch_status": fetch_status,
            "snapshot_hash": "",
            "error_type": error_type,
            "error_message": error_message
        }


def fetch_source_content(url: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Convenience function for single URL fetching.

    Args:
        url: Target URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Source fetching result dict
    """
    fetcher = SourceFetcher(timeout=timeout)
    return fetcher.fetch_and_extract(url)


if __name__ == "__main__":
    # Test the source fetcher
    import sys

    logging.basicConfig(level=logging.INFO)

    test_urls = [
        "https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance",
        "https://www.cdc.gov/tb/causes/index.html",
        "https://www.example-invalid-url-12345.com",
        "not-a-url"
    ]

    for url in test_urls:
        print(f"\n🧪 Testing: {url}")
        print("=" * 60)

        result = fetch_source_content(url)

        print(f"Availability Signal: {result['availability_signal']}")
        print(f"Content Length: {result['content_length']}")
        print(f"Fetch Time: {result['fetch_time_ms']}ms")
        print(f"Final URL: {result['final_url']}")

        if result['error_type']:
            print(f"Error Type: {result['error_type']}")
            print(f"Error Message: {result['error_message']}")
        else:
            preview_text = result['extracted_text'][:200] + "..." if len(result['extracted_text']) > 200 else result['extracted_text']
            print(f"Text Preview: {preview_text}")
