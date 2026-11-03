"""Domain errors. The API layer maps each to an HTTP status; nothing else catches them."""


class FastRagError(Exception):
    """Base class for expected, user-facing errors."""


class AuthenticationError(FastRagError):
    """Missing or invalid credentials (HTTP 401)."""


class PermissionDenied(FastRagError):
    """Authenticated, but not allowed to do this (HTTP 403)."""


class NotFound(FastRagError):
    """The resource doesn't exist, or the caller may not know it exists (HTTP 404)."""


class RateLimited(FastRagError):
    """Too many requests for this tenant (HTTP 429)."""

    def __init__(self, retry_after: float):
        super().__init__(f"Rate limit exceeded; retry after {retry_after:.1f}s")
        self.retry_after = retry_after


class ConcurrentModification(FastRagError):
    """The document changed between reading and writing it (HTTP 409).

    Raised by stores when a write's expected version no longer matches; the ingest
    service retries with fresh state, so callers only see it under heavy contention.
    """


class InvalidInput(FastRagError):
    """The request is well-formed but can't be processed (HTTP 422)."""


class GenerationError(FastRagError):
    """The answer generator failed or declined (HTTP 502)."""


class ConfigurationError(FastRagError):
    """The system is misconfigured (fails fast at startup)."""
