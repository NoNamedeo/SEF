from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from library.core.abstractions.ISignal import ISignal
from library.core.artifacts.SignalSample import SignalSample


class Signal(ISignal):
    """Concrete signal container used across the library."""

    def __init__(
        self,
        samples: Sequence[SignalSample],
        config: dict[str, Any] | None = None,
    ):
        super().__init__(samples, config)
