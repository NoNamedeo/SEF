from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from library.core.events.Event import Event

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
    def subscribe(self, event_type: str, handler: EventHandler) -> None: ...

    @abstractmethod
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None: ...

    @abstractmethod
    def dispatch(self, event: Event) -> None: ...

    @abstractmethod
    async def publish(self, event: Event) -> None: ...
