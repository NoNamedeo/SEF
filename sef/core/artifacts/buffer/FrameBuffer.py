from __future__ import annotations

from collections.abc import Iterable, Iterator
from queue import Empty, Full, Queue
from time import sleep

from sef.core.artifacts.Frame import Frame

_SENTINEL = object()


class FrameBuffer:
    """
    Thread-safe frame buffer for batch and streaming execution.

    `FrameBuffer` exposes a public capacity for frame items and reserves one
    internal queue slot for the end-of-stream sentinel. Iteration blocks until
    frames are available or the buffer is closed.

    Ordering
    --------
    Frames are yielded in the order accepted by `put()` or `try_put()`.

    Thread safety
    -------------
    The queue operations are thread-safe. The `Frame` objects themselves are
    not copied; producers and consumers remain responsible for pixel mutability.
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
        """
        Publish a frame, blocking while the public queue is full.

        The method returns silently when the buffer has already been closed.
        """
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
        """
        Return the next frame or raise `StopIteration` when closed.

        The end-of-stream sentinel is reinserted so repeated consumers observe
        completion consistently.
        """
        item = self._queue.get()

        if item is _SENTINEL:
            # Reinsert the sentinel so other consumers also observe completion.
            self._queue.put(_SENTINEL)
            self.closed = True
            raise StopIteration

        return item

    def close(self) -> None:
        """Close the buffer and wake consumers waiting for more frames."""
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
        """Return whether no frame or sentinel item is currently queued."""
        return self._queue.empty()

    def size(self) -> int:
        """Return current queue size, including a sentinel if present."""
        return self._queue.qsize()

    def fill_ratio(self) -> float:
        """Return current public queue occupancy in the [0, 1] range."""
        if self.capacity <= 0:
            return 0.0
        return min(1.0, max(0.0, self.size() / self.capacity))

    def clone_empty(self) -> "FrameBuffer":
        """Return an empty buffer with the same public capacity."""
        return FrameBuffer(buffer_size=self.capacity)

    def __iter__(self) -> Iterator[Frame]:
        """Iterate frames until the buffer closes."""
        while True:
            try:
                yield self.get()
            except StopIteration:
                break
