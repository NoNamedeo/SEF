from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalSample import ISignalSample


class Signal(ISignal):
    """Concrete signal container used across the sef.builtin."""

    def __init__(
        self,
        samples: Sequence[ISignalSample],
        config: dict[str, Any] | None = None,
    ):
        super().__init__(samples, config)
