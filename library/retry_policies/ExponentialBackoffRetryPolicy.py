from __future__ import annotations

from library.core.abstractions.IRetryPolicy import IRetryPolicy


class ExponentialBackoffRetryPolicy(IRetryPolicy):
    """
    Retries with exponentially increasing delays between attempts.

    Wait time formula: ``min(base_delay * 2^(attempt - 1), max_delay)``

    Parameters
    ----------
    max_retries:
        Maximum number of *additional* attempts after the first failure.
        Must be >= 1.
    base_delay:
        Seconds to wait before the *second* attempt (after the 1st failure).
        Each subsequent wait doubles. Defaults to ``1.0`` s.
    max_delay:
        Upper bound on the computed wait, to avoid unbounded sleeps.
        Defaults to ``60.0`` s.

    Example
    -------
    >>> policy = ExponentialBackoffRetryPolicy(max_retries=4, base_delay=2.0)
    >>> policy.wait_seconds(1)   # 2.0 s  (before 2nd attempt)
    >>> policy.wait_seconds(2)   # 4.0 s  (before 3rd attempt)
    >>> policy.wait_seconds(3)   # 8.0 s  (before 4th attempt)
    """

    def __init__(
        self,
        max_retries: int,
        base_delay:  float = 1.0,
        max_delay:   float = 60.0,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1.")
        if base_delay <= 0:
            raise ValueError("base_delay must be > 0.")
        if max_delay < base_delay:
            raise ValueError("max_delay must be >= base_delay.")
        self._max_retries = max_retries
        self._base_delay  = base_delay
        self._max_delay   = max_delay

    def should_retry(self, attempt: int, error: Exception) -> bool:
        return attempt <= self._max_retries

    def wait_seconds(self, attempt: int) -> float:
        delay = self._base_delay * (2 ** (attempt - 1))
        return min(delay, self._max_delay)
