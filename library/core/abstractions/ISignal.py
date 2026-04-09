from __future__ import annotations

from abc import ABC
from collections.abc import Iterator, Sequence
from typing import Any

from library.core.artifacts.SignalSample import SignalSample


class ISignal(ABC):
    """Base contract for extracted signals."""

    def __init__(
        self,
        samples: Sequence[SignalSample],
        config: dict[str, Any] | None = None,
    ):
        self.samples = list(samples)
        self.config = config or {}

    @property
    def signal(self) -> list[SignalSample]:
        """Backward-compatible alias for existing code."""
        return self.samples

    def __iter__(self) -> Iterator[SignalSample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)
