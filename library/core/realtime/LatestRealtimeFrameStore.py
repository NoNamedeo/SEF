from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from library.core.realtime.IRealtimeFrameSink import IRealtimeFrameSink
from library.core.realtime.RealtimeFrame import RealtimeFrame


@dataclass(frozen=True, slots=True)
class RealtimeFrameSnapshot:
    """Point-in-time view of a realtime frame store."""

    frame: RealtimeFrame | None
    version: int
    active: bool
    updated_at: datetime | None
    published_frames: int
    last_stage: str | None


class LatestRealtimeFrameStore(IRealtimeFrameSink):
    """
    Thread-safe sink that keeps only the newest realtime frame.

    This implements a drop-oldest preview policy: UI consumers never block the
    pipeline, and slow refresh rates simply skip stale frames.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: RealtimeFrame | None = None
        self._version = 0
        self._published_frames = 0
        self._last_stage: str | None = None
        self._active = False
        self._updated_at: datetime | None = None

    def publish(self, frame: RealtimeFrame) -> None:
        with self._lock:
            if self._should_keep_current_frame(frame):
                return
            self._frame = RealtimeFrame(
                image=frame.image.copy(),
                color_space=frame.color_space,
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                produced_at=frame.produced_at,
                metadata=frame.metadata,
            )
            self._version += 1
            self._published_frames += 1
            self._last_stage = str(frame.metadata.get("preview_stage", "unknown"))
            self._active = True
            self._updated_at = datetime.now(timezone.utc)

    def close(self) -> None:
        with self._lock:
            self._active = False
            self._updated_at = datetime.now(timezone.utc)

    def reset(self) -> None:
        """Clear any previous preview before a new run starts."""
        with self._lock:
            self._frame = None
            self._version = 0
            self._published_frames = 0
            self._last_stage = None
            self._active = True
            self._updated_at = datetime.now(timezone.utc)

    def snapshot(self) -> RealtimeFrameSnapshot:
        """Return a copy-safe snapshot for UI rendering."""
        with self._lock:
            frame = None
            if self._frame is not None:
                frame = RealtimeFrame(
                    image=self._frame.image.copy(),
                    color_space=self._frame.color_space,
                    frame_index=self._frame.frame_index,
                    timestamp_seconds=self._frame.timestamp_seconds,
                    produced_at=self._frame.produced_at,
                    metadata=self._frame.metadata,
                )
            return RealtimeFrameSnapshot(
                frame=frame,
                version=self._version,
                active=self._active,
                updated_at=self._updated_at,
                published_frames=self._published_frames,
                last_stage=self._last_stage,
            )

    def _should_keep_current_frame(self, incoming: RealtimeFrame) -> bool:
        """
        Keep annotated previews stable once they start arriving.

        Raw frame taps are useful as an initial liveness signal, but they run at
        camera FPS while pose inference is slower. Replacing annotated frames
        with newer raw frames makes the UI flicker between unrelated layers.
        """
        if self._frame is None:
            return False
        current_priority = _preview_priority(self._frame)
        incoming_priority = _preview_priority(incoming)
        return incoming_priority < current_priority


def _preview_priority(frame: RealtimeFrame) -> int:
    raw_priority = frame.metadata.get("preview_priority")
    if raw_priority is not None:
        return int(raw_priority)
    stage = str(frame.metadata.get("preview_stage", ""))
    if stage == "coco_pose_visualizer":
        return 100
    if stage == "frame_tap":
        return 10
    return 0
