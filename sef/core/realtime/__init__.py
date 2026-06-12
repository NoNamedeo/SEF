"""Realtime preview contracts for UI and service adapters.

The realtime package decouples live frame producers from viewers. Visualizers
or processors publish immutable `RealtimeFrame` values to an `IRealtimeFrameSink`;
adapters decide how often to poll, encode, stream, or display the latest frame.

The core contract intentionally avoids Streamlit, OpenCV windows, web sockets,
or HTTP dependencies so the same producer can feed multiple presentation
strategies.
"""

from sef.core.realtime.IRealtimeFrameSink import IRealtimeFrameSink
from sef.core.realtime.LatestRealtimeFrameStore import LatestRealtimeFrameStore, RealtimeFrameSnapshot
from sef.core.realtime.NullRealtimeFrameSink import NullRealtimeFrameSink
from sef.core.realtime.RealtimeFrame import FrameColorSpace, RealtimeFrame

__all__ = [
    "FrameColorSpace",
    "IRealtimeFrameSink",
    "LatestRealtimeFrameStore",
    "NullRealtimeFrameSink",
    "RealtimeFrame",
    "RealtimeFrameSnapshot",
]
