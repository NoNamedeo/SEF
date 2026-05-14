from __future__ import annotations

from collections.abc import Iterable
from threading import Condition
from typing import Dict, List

from library.core.interfaces.IData import IData


_SENTINEL = object()


class DataBuffer:
    """
    Multi-consumer streaming buffer with reference counting.
    """

    def __init__(
        self,
        buffer_size: int | None = None,
        data: Iterable[IData] | None = None,
        consumers: int = 1,
    ):
        self.capacity = buffer_size or 10

        self._cond = Condition()
        self._data: List[IData | object] = []
        self._refcounts: List[int] = []

        self._consumers_default = consumers
        self._subscribers: Dict[int, int] = {}

        self._closed = False

        if data:
            for d in data:
                self.put(d)

    def put(self, item: IData) -> None:
        with self._cond:
            while self.capacity > 0 and len(self._data) >= self.capacity:
                self._cond.wait()

            self._data.append(item)
            self._refcounts.append(self._consumers_default)
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._data.append(_SENTINEL)
            self._refcounts.append(self._consumers_default)
            self._cond.notify_all()

    def subscribe(self, consumer_id: int):
        with self._cond:
            self._subscribers[consumer_id] = 0
            return DataSubscription(self, consumer_id)

    def _get_for(self, consumer_id: int) -> IData:
        with self._cond:
            idx = self._subscribers[consumer_id]

            while idx >= len(self._data):
                if self._closed:
                    raise StopIteration
                self._cond.wait()

            item = self._data[idx]

            if item is _SENTINEL:
                raise StopIteration

            self._refcounts[idx] -= 1

            if self._refcounts[idx] <= 0:
                self._data[idx] = None
                self._refcounts[idx] = 0
                self._compact_front()

            self._subscribers[consumer_id] += 1
            self._cond.notify_all()

            return item

    def _compact_front(self):
        while self._refcounts and self._refcounts[0] == 0:
            self._refcounts.pop(0)
            self._data.pop(0)
            for k in self._subscribers:
                self._subscribers[k] = max(0, self._subscribers[k] - 1)


class DataSubscription:
    def __init__(self, buffer: DataBuffer, consumer_id: int):
        self._buffer = buffer
        self._id = consumer_id

    def __iter__(self):
        return self

    def __next__(self):
        return self._buffer._get_for(self._id)