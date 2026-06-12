from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, TypeVar, runtime_checkable

from library.core.artifacts.Frame import Frame

T = TypeVar("T")


@runtime_checkable
class IBuffer(Protocol[T]):
    """Minimal producer-side contract shared by pipeline buffers."""

    @property
    def closed(self) -> bool:
        """Return True when the buffer no longer accepts new items."""
        ...

    def put(self, item: T) -> None:
        """Publish one item into the buffer."""
        ...

    def close(self) -> None:
        """Mark the stream as complete and wake consumers."""
        ...


@runtime_checkable
class IAbortableBuffer(IBuffer[T], Protocol[T]):
    """Buffer that can interrupt a stream immediately."""

    def abort(self) -> None:
        """Cancel the stream and unblock waiting producers or consumers."""
        ...


@runtime_checkable
class IBufferSubscription(Iterator[T], Protocol[T]):
    """Consumer-side stream cursor that can cancel its upstream buffer."""

    def abort(self) -> None:
        """Cancel the source buffer for cooperative downstream shutdown."""
        ...


@runtime_checkable
class ISubscribableBuffer(IAbortableBuffer[T], Protocol[T]):
    """Multi-consumer buffer with explicit fan-out configuration."""

    def set_consumer_count(self, consumers: int) -> None:
        """Declare how many consumers must observe each future item."""
        ...

    def subscribe(self, consumer_id: int) -> IBufferSubscription[T]:
        """Return a consumer cursor for the configured stream."""
        ...


@runtime_checkable
class IFrameBuffer(IAbortableBuffer[Frame], Iterable[Frame], Protocol):
    """
    Frame stream contract required by realtime latency policies.

    This intentionally lives outside ``IBuffer`` because capacity and dropping
    are frame-queue concerns, not universal buffer responsibilities.
    """

    @property
    def capacity(self) -> int:
        """Maximum number of frame items retained by the public queue."""
        ...

    def try_put(self, frame: Frame) -> bool:
        """Publish without blocking. Return False when the frame is rejected."""
        ...

    def drop_oldest(self) -> Frame | None:
        """Drop and return the oldest queued frame item, when available."""
        ...

    def fill_ratio(self) -> float:
        """Return public queue occupancy in the [0, 1] range."""
        ...
