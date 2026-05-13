from __future__ import annotations

from typing import Any
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.BoxSignalSample import BoundingBox, BoxSignalSample


class NoSignalExtractor(ISignalExtractor):
    """No operation tracker."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

    def extract(self, buffer: FrameBuffer) -> ISignal:
        for frame in buffer:
            pass
        return Signal([BoxSignalSample(0,(0,0,0,0),(0,0))])

