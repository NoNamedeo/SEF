from __future__ import annotations

from sef.core.interfaces.pipeline.IRetryPolicy import IRetryPolicy


class FixedRetryPolicy(IRetryPolicy):
    """
    Retries up to *max_retries* times with no delay between attempts.

    Parameters
    ----------
    max_retries:
        Maximum number of *additional* attempts after the first failure.
        Must be >= 1 (use ``NoRetryPolicy`` when you want zero retries).

    Example
    -------
    >>> policy = FixedRetryPolicy(max_retries=3)
    >>> policy.should_retry(1, exc)   # True  – first failure, retry allowed
    >>> policy.should_retry(3, exc)   # True  – third failure, last retry
    >>> policy.should_retry(4, exc)   # False – exhausted
    """

    def __init__(self, max_retries: int) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1. Use NoRetryPolicy for zero retries.")
        self._max_retries = max_retries

    def should_retry(self, attempt: int, error: Exception) -> bool:
        return attempt <= self._max_retries
