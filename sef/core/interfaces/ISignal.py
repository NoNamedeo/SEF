from __future__ import annotations

from abc import ABC
from collections.abc import Iterator, Sequence
from typing import Any

from sef.core.interfaces.ISignalSample import ISignalSample


class ISignal(ABC):
    """
    Base contract for ordered signal samples.

    Signals are analyzer input values. Batch signal extractors return this
    complete collection, while streaming extractors publish `ISignalSample`
    values progressively into a buffer.
    """

    def __init__(
        self,
        samples: Sequence[ISignalSample],
        config: dict[str, Any] | None = None,
    ):
        """
        Create a signal from ordered samples.

        Parameters
        ----------
        samples:
            Signal samples in analysis order.
        config:
            Optional plugin-specific metadata retained for compatibility.
        """
        self.samples = list(samples)
        self.config = config or {}

    @property
    def signal(self) -> list[ISignalSample]:
        """Backward-compatible alias for `samples`."""
        return self.samples

    def __iter__(self) -> Iterator[ISignalSample]:
        """Iterate samples in analysis order."""
        return iter(self.samples)

    def __len__(self) -> int:
        """Return the number of samples in the signal."""
        return len(self.samples)
