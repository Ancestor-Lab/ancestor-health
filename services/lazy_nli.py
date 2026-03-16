from functools import lru_cache
@lru_cache(maxsize=1)
def get_nli_service():
    """Lazy loading of NLI service to avoid startup timeout."""
    from services.nli_service import NLIService
    return NLIService()
