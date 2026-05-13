from __future__ import annotations

from collections.abc import Iterable, Iterator
from queue import Queue

from library.core.interfaces.ISignalSample import ISignalSample


_SENTINEL = object()


class SignalBuffer:
    """
    Thread-safe streaming buffer for signal samples.
    """

    def __init__(
        self,
        buffer_size: int | None = None,
        samples: Iterable[ISignalSample] | None = None,
    ):
        self.capacity = buffer_size or 0
        self._queue: Queue = Queue(maxsize=self.capacity)
        self.closed = False

        if samples:
            for s in samples:
                self.put(s)

    def put(self, sample: ISignalSample) -> None:
        self._queue.put(sample)

    def get(self) -> ISignalSample:
        item = self._queue.get()

        if item is _SENTINEL:
            self._queue.put(_SENTINEL)
            self.closed = True
            raise StopIteration

        return item

    def close(self) -> None:
        self.closed = True
        self._queue.put(_SENTINEL)

    def is_empty(self) -> bool:
        return self._queue.empty()

    def size(self) -> int:
        return self._queue.qsize()

    def clone_empty(self) -> "SignalBuffer":
        return SignalBuffer(buffer_size=self.capacity)

    def __iter__(self) -> Iterator[ISignalSample]:
        while True:
            try:
                yield self.get()
            except StopIteration:
                break