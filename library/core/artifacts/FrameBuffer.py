from __future__ import annotations

from collections.abc import Iterable, Iterator
from queue import Empty, Full, Queue
from time import sleep

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
        self.capacity = buffer_size or 10
        # Keep one private slot for the end-of-stream sentinel. Public capacity
        # still describes the maximum number of frame items.
        self._queue: Queue = Queue(maxsize=self.capacity + 1)
        self.closed = False

        if frames:
            for f in frames:
                self.put(f)

    def put(self, frame: Frame) -> None:
        while not self.closed:
            try:
                self._queue.put(frame, timeout=0.05)
                return
            except Full:
                continue

    def try_put(self, frame: Frame) -> bool:
        """Publish a frame without blocking. Return False when the queue is full."""
        if self.closed or self.size() >= self.capacity:
            return False
        try:
            self._queue.put_nowait(frame)
            return True
        except Full:
            return False

    def drop_oldest(self) -> Frame | None:
        """Drop and return the oldest queued frame item, if one is available."""
        if self.closed:
            return None
        try:
            item = self._queue.get_nowait()
        except Empty:
            return None
        if item is _SENTINEL:
            self._queue.put(_SENTINEL)
            return None
        return item

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
        while True:
            try:
                self._queue.put_nowait(_SENTINEL)
                return
            except Full:
                try:
                    self._queue.get_nowait()
                except Empty:
                    sleep(0.001)

    def abort(self) -> None:
        """Close the buffer without blocking, dropping queued frames if needed."""
        self.closed = True
        while True:
            try:
                self._queue.put_nowait(_SENTINEL)
                return
            except Full:
                try:
                    self._queue.get_nowait()
                except Empty:
                    continue

    def is_empty(self) -> bool:
        return self._queue.empty()

    def size(self) -> int:
        return self._queue.qsize()

    def fill_ratio(self) -> float:
        """Return current public queue occupancy in the [0, 1] range."""
        if self.capacity <= 0:
            return 0.0
        return min(1.0, max(0.0, self.size() / self.capacity))

    def clone_empty(self) -> "FrameBuffer":
        return FrameBuffer(buffer_size=self.capacity)

    def __iter__(self) -> Iterator[Frame]:
        while True:
            try:
                yield self.get()
            except StopIteration:
                break
