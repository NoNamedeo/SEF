from __future__ import annotations

from collections.abc import Iterable, Iterator
from queue import Queue

from library.core.interfaces.IData import IData


_SENTINEL = object()


class DataBuffer(IData):
    """
    Thread-safe streaming buffer for analysis outputs.
    """

    def __init__(
        self,
        buffer_size: int | None = None,
        data: Iterable[IData] | None = None,
    ):
        self.capacity = buffer_size or 0
        self._queue: Queue = Queue(maxsize=self.capacity)
        self.closed = False

        if data:
            for d in data:
                self.put(d)

    def put(self, item: IData) -> None:
        self._queue.put(item)

    def get(self) -> IData:
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

    def clone_empty(self) -> "DataBuffer":
        return DataBuffer(buffer_size=self.capacity)

    def __iter__(self) -> Iterator[IData]:
        while True:
            try:
                yield self.get()
            except StopIteration:
                break