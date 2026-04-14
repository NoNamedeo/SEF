from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


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

    @abstractmethod
    def subscribe(self, event_type: str, handler: Callable[[Any], None]) -> None: ...

    @abstractmethod
    def unsubscribe(self, event_type: str, handler: Callable[[Any], None]) -> None: ...

    @abstractmethod
    def dispatch(self, event: Any) -> None: ...

    @abstractmethod
    async def publish(self, event: Any) -> None: ...
