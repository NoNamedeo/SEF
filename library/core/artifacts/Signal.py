from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from library.core.abstractions.ISignal import ISignal
from library.core.abstractions.ISignalSample import ISignalSample
from library.core.artifacts.SignalSample import SignalSample
from library.core.interfaces.ISignal import ISignal


class Signal(ISignal):
    """Concrete signal container used across the library."""

    def __init__(
        self,
        samples: Sequence[ISignalSample],
        config: dict[str, Any] | None = None,
    ):
        super().__init__(samples, config)
