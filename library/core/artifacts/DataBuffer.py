from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

from library.core.interfaces.IData import IData


class DataBuffer(IData):
    """
    In-memory handoff for analytical data between stream analyzers and visualizers.

    The buffer mirrors FrameBuffer semantics to keep stream pipeline stages
    consistent from frames to signals to chart-ready data.
    """

    def __init__(self, buffer_size: int | None = None, data: Iterable[IData] | None = None):
        self.capacity = buffer_size
        self._data = deque(data or [])
        self.closed = False

    def put(self, item: IData) -> None:
        self._data.append(item)

    def get(self) -> IData:
        if self.is_empty():
            raise IndexError("DataBuffer is empty")
        return self._data.popleft()

    def close(self) -> None:
        self.closed = True

    def is_empty(self) -> bool:
        return not self._data

    def size(self) -> int:
        return len(self._data)

    def clone_empty(self) -> "DataBuffer":
        return DataBuffer(buffer_size=self.capacity)

    def __iter__(self) -> Iterator[IData]:
        while not self.closed or self._data:
            if self._data:
                yield self.get()

