from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.interfaces.ISignal import ISignal

ThreadedStageTask = Callable[[ThreadPoolExecutor], Future[Any]]


@dataclass
class FrameRuntimeState:
    """Mutable runtime state for the current frame segment."""

    buffer: FrameBuffer
    pending_tasks: list[ThreadedStageTask] = field(default_factory=list)
    buffers: list[FrameBuffer] = field(default_factory=list)

    @property
    def is_streaming(self) -> bool:
        return bool(self.pending_tasks)


@dataclass
class SignalRuntimeState:
    """Mutable runtime state for the current signal segment."""

    signal: ISignal | None = None
    buffer: SignalBuffer | None = None
    pending_tasks: list[ThreadedStageTask] = field(default_factory=list)
    buffers: list[Any] = field(default_factory=list)

    @property
    def is_streaming(self) -> bool:
        return self.buffer is not None and bool(self.pending_tasks)
