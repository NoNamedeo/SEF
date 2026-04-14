from __future__ import annotations

from library.core.interfaces.pipeline.IRetryPolicy import IRetryPolicy


class NoRetryPolicy(IRetryPolicy):
    """
    Disables retries entirely — the orchestrator re-raises on the first failure.

    This is the sensible default: explicit opt-in to retries is safer than
    silent swallowing of errors.
    """

    def should_retry(self, attempt: int, error: Exception) -> bool:
        return False
