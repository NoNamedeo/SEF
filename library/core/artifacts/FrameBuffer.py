from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

from library.core.artifacts.Frame import Frame


class FrameBuffer:
    """
    In-memory frame handoff between extractors and signal extractors.

    the pipeline is synchronous. This buffer is
    intentionally simple and deterministic so the core can be executed
    end-to-end without extra threading infrastructure.
    """

    def __init__(self, buffer_size: int | None = None, frames: Iterable[Frame] | None = None):
        self.capacity = buffer_size
        self._frames = deque(frames or [])
        self.closed = False

    def put(self, frame: Frame) -> None:
        self._frames.append(frame)

    def get(self) -> Frame:
        if self.is_empty():
            raise IndexError("FrameBuffer is empty")
        return self._frames.popleft()

    def close(self) -> None:
        self.closed = True

    def is_empty(self) -> bool:
        return not self._frames

    def size(self) -> int:
        return len(self._frames)

    def clone_empty(self) -> "FrameBuffer":
        return FrameBuffer(buffer_size=self.capacity)

    def __iter__(self) -> Iterator[Frame]:
        while not self.closed or self._frames:
            if self._frames:
                yield self.get()
