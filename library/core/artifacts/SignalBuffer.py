from __future__ import annotations

from collections.abc import Iterable
from threading import Condition
from typing import Dict, List

from library.core.interfaces.BufferContracts import IBufferSubscription
from library.core.interfaces.ISignalSample import ISignalSample

_SENTINEL = object()


class SignalBuffer:
    """
    Multi-consumer streaming buffer with reference counting.
    """

    def __init__(
        self,
        buffer_size: int | None = None,
        samples: Iterable[ISignalSample] | None = None,
        consumers: int = 1,
    ):
        self.capacity = buffer_size or 10

        self._cond = Condition()
        self._data: List[ISignalSample | object] = []
        self._refcounts: List[int] = []

        self._consumers_default = consumers

        self._subscribers: Dict[int, int] = {}  # consumer_id -> index

        self._closed = False

        if samples:
            for s in samples:
                self.put(s)

    # -------------------------
    # PRODUCER
    # -------------------------
    def put(self, sample: ISignalSample) -> None:
        with self._cond:
            if self._closed or self._consumers_default <= 0:
                return

            while self.capacity > 0 and len(self._data) >= self.capacity:
                if self._closed:
                    return
                self._cond.wait()

            self._data.append(sample)
            self._refcounts.append(self._consumers_default)
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        """Return True when the buffer has been closed or aborted."""
        with self._cond:
            return self._closed

    def close(self) -> None:
        with self._cond:
            self._closed = True
            if self._consumers_default <= 0:
                self._cond.notify_all()
                return
            self._data.append(_SENTINEL)
            self._refcounts.append(self._consumers_default)
            self._cond.notify_all()

    def abort(self) -> None:
        """Wake all consumers and discard buffered samples after an upstream failure."""
        with self._cond:
            self._closed = True
            self._data.clear()
            self._refcounts.clear()
            self._cond.notify_all()

    def set_consumer_count(self, consumers: int) -> None:
        """Declare how many subscribers must consume each future sample."""
        if consumers < 0:
            raise ValueError("SignalBuffer consumers cannot be negative.")
        with self._cond:
            if self._data:
                raise RuntimeError("SignalBuffer consumer count must be configured before samples are produced.")
            self._consumers_default = consumers

    # -------------------------
    # SUBSCRIPTION
    # -------------------------
    def subscribe(self, consumer_id: int) -> IBufferSubscription[ISignalSample]:
        with self._cond:
            self._subscribers[consumer_id] = 0
            return SignalSubscription(self, consumer_id)

    # -------------------------
    # INTERNAL GET LOGIC
    # -------------------------
    def _get_for(self, consumer_id: int) -> ISignalSample:
        with self._cond:
            idx = self._subscribers[consumer_id]

            while idx >= len(self._data):
                if self._closed:
                    raise StopIteration
                self._cond.wait()

            item = self._data[idx]

            if item is _SENTINEL:
                raise StopIteration

            self._subscribers[consumer_id] += 1
            self._refcounts[idx] -= 1

            if self._refcounts[idx] <= 0:
                # cleanup safe (all consumers have seen it)
                self._data[idx] = None  # free memory
                self._refcounts[idx] = 0

                # advance global cleanup window
                self._compact_front()

            self._cond.notify_all()
            return item

    def _compact_front(self):
        # remove only safe prefix
        while self._refcounts and self._refcounts[0] == 0:
            self._refcounts.pop(0)
            self._data.pop(0)
            for k in self._subscribers:
                self._subscribers[k] = max(0, self._subscribers[k] - 1)


class SignalSubscription:
    def __init__(self, buffer: SignalBuffer, consumer_id: int):
        self._buffer = buffer
        self._id = consumer_id

    def __iter__(self):
        return self

    def __next__(self) -> ISignalSample:
        return self._buffer._get_for(self._id)

    def abort(self) -> None:
        """Abort the source buffer for cooperative downstream cancellation."""
        self._buffer.abort()
