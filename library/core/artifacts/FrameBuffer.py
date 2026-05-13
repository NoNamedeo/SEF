from __future__ import annotations

from collections.abc import Iterable, Iterator
from queue import Queue

from library.core.artifacts.Frame import Frame


_SENTINEL = object()


class FrameBuffer:
    """
    Thread-safe streaming buffer based on Queue.
    """

    def __init__(
        self,
        buffer_size: int | None = None,
        frames: Iterable[Frame] | None = None,
    ):
        self.capacity = buffer_size or 0
        self._queue: Queue = Queue(maxsize=self.capacity)
        self.closed = False

        if frames:
            for f in frames:
                self.put(f)

    def put(self, frame: Frame) -> None:
        self._queue.put(frame)

    def get(self) -> Frame:
        item = self._queue.get()

        if item is _SENTINEL:
            # ripubblica sentinel per altri consumer
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

    def clone_empty(self) -> "FrameBuffer":
        return FrameBuffer(buffer_size=self.capacity)

    def __iter__(self) -> Iterator[Frame]:
        while True:
            try:
                yield self.get()
            except StopIteration:
                break