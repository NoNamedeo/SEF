from __future__ import annotations

from sef.core.interfaces.pipeline.IRetryPolicy import IRetryPolicy


class NoRetryPolicy(IRetryPolicy):
    """
    Disable retries entirely.

    This is the runner default because retrying arbitrary video pipelines can
    hide component bugs or repeat expensive side effects. Applications that
    want retries can inject an explicit ``IRetryPolicy`` implementation.
    """

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Return ``False`` for every failure."""
        return False
