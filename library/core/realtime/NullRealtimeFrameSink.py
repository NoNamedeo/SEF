from __future__ import annotations

from library.core.realtime.IRealtimeFrameSink import IRealtimeFrameSink
from library.core.realtime.RealtimeFrame import RealtimeFrame


class NullRealtimeFrameSink(IRealtimeFrameSink):
    """Sink used when realtime rendering is configured without an external consumer."""

    def publish(self, frame: RealtimeFrame) -> None:
        return

    def close(self) -> None:
        return
