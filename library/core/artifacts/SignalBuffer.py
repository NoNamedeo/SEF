from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

from library.core.interfaces.ISignalSample import ISignalSample


class SignalBuffer:
    """
    In-memory handoff for signal samples between stream extractors and analyzers.

    This mirrors FrameBuffer semantics so stream-oriented components can
    exchange samples incrementally through a simple synchronous queue.
    """

    def __init__(self, buffer_size: int | None = None, samples: Iterable[ISignalSample] | None = None):
        self.capacity = buffer_size
        self._samples = deque(samples or [])
        self.closed = False

    def put(self, sample: ISignalSample) -> None:
        self._samples.append(sample)

    def get(self) -> ISignalSample:
        if self.is_empty():
            raise IndexError("SignalBuffer is empty")
        return self._samples.popleft()

    def close(self) -> None:
        self.closed = True

    def is_empty(self) -> bool:
        return not self._samples

    def size(self) -> int:
        return len(self._samples)

    def clone_empty(self) -> "SignalBuffer":
        return SignalBuffer(buffer_size=self.capacity)

    def __iter__(self) -> Iterator[ISignalSample]:
        while not self.closed or self._samples:
            if self._samples:
                yield self.get()

