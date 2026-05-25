from __future__ import annotations

from typing import Protocol

from library.core.realtime.RealtimeFrame import RealtimeFrame


class IRealtimeFrameSink(Protocol):
    """Output port for realtime visualization frames."""

    def publish(self, frame: RealtimeFrame) -> None:
        """Publish a rendered frame to a realtime consumer."""

    def close(self) -> None:
        """Notify consumers that no more frames will be published."""
