from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from library.core.artifacts.buffer.FrameBuffer import FrameBuffer
from library.core.interfaces.BufferContracts import IAbortableBuffer, ISubscribableBuffer
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalSample import ISignalSample

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
    buffer: ISubscribableBuffer[ISignalSample] | None = None
    pending_tasks: list[ThreadedStageTask] = field(default_factory=list)
    buffers: list[IAbortableBuffer[Any]] = field(default_factory=list)

    @property
    def is_streaming(self) -> bool:
        return self.buffer is not None and bool(self.pending_tasks)
