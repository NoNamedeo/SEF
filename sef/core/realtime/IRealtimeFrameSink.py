from __future__ import annotations

from typing import Protocol

from sef.core.realtime.RealtimeFrame import RealtimeFrame


class IRealtimeFrameSink(Protocol):
    """
    Output port for realtime visualization frames.

    Sinks decouple frame producers from UI refresh loops. Implementations should
    avoid blocking the pipeline for slow viewers; copy frames when crossing
    thread or process boundaries.
    """

    def publish(self, frame: RealtimeFrame) -> None:
        """Publish a rendered frame to a realtime consumer."""

    def close(self) -> None:
        """Notify consumers that no more frames will be published."""
