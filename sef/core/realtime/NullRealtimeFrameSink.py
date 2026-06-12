from __future__ import annotations

from sef.core.realtime.IRealtimeFrameSink import IRealtimeFrameSink
from sef.core.realtime.RealtimeFrame import RealtimeFrame


class NullRealtimeFrameSink(IRealtimeFrameSink):
    """
    No-op realtime sink.

    Use this sink when a component accepts realtime publication but the current
    application has no viewer attached.
    """

    def publish(self, frame: RealtimeFrame) -> None:
        return

    def close(self) -> None:
        return
