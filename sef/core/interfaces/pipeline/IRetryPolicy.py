from __future__ import annotations

from abc import ABC, abstractmethod


class IRetryPolicy(ABC):
    """
    Strategy interface that governs retry behavior for runners.

    Design rationale
    ----------------
    Extracting the retry decision into its own object follows the
    Open/Closed Principle: new strategies such as backoff, jitter, or circuit
    breakers are added by creating a new class, never by editing the runner.
    The runner calls only two methods:

    - `should_retry`: binary gate deciding whether to keep trying.
    - `wait_seconds`: optional pause before the next attempt.

    Both receive the 1-based attempt number that just failed, so policies
    can implement attempt-dependent logic such as exponential backoff.
    """

    @abstractmethod
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """
        Decide whether execution should be retried.

        Parameters
        ----------
        attempt:
            The 1-based number of the attempt that *just failed*.
        error:
            The exception that caused the failure.

        Returns
        -------
        bool
            `True` when the runner should make another attempt. `False` when
            the runner should stop and re-raise `error`.
        """

    def wait_seconds(self, attempt: int) -> float:
        """
        Seconds to pause *before* the next attempt.

        Override in subclasses that implement backoff strategies.
        The base implementation returns ``0.0`` (no wait).

        Parameters
        ----------
        attempt:
            The 1-based number of the attempt that *just failed*.
        """
        return 0.0
