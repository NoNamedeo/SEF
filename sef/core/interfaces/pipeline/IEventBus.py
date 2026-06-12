from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from sef.core.events.Event import Event

EventHandler = Callable[[Event], None]


class IEventBus(ABC):
    """
    Typed async-first pub/sub interface.

    Two publication paths
    --------------------
    * ``dispatch`` — synchronous, for use within pipeline execution threads
      where no event loop is running.
    * ``publish``  — async, for external callers operating in an asyncio
      context.

    Per-listener error isolation: a failing handler must never prevent other
    handlers from receiving the same event.
    """

    WILDCARD = "*"

    @abstractmethod
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Register a handler for one event type.

        Use `IEventBus.WILDCARD` to observe every event.
        """
        ...

    @abstractmethod
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a previously registered handler when present."""
        ...

    @abstractmethod
    def dispatch(self, event: Event) -> None:
        """
        Publish an event synchronously.

        Implementations should isolate handler failures so one subscriber does
        not prevent remaining subscribers from receiving the event.
        """
        ...

    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish an event from an async caller."""
        ...
