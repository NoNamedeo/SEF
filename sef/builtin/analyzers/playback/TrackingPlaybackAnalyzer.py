from __future__ import annotations

from typing import Any

from library.core.artifacts.signal_sample.BoxSignalSample import BoxSignalSample
from library.core.artifacts.signal_sample.MultiObjectSignalSample import (
    MultiObjectSignalSample,
    MultiObjectTrack,
)
from library.core.artifacts.data.TrackingPlaybackData import (
    TrackingPlaybackData,
    TrackingPlaybackFrame,
    TrackingPlaybackTrack,
)
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalSample import ISignalSample


class TrackingPlaybackAnalyzer(IAnalyzer):
    """Convert tracker samples into playback-ready frame annotations."""

    DEFAULT_TITLE = "Tracking Playback"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._title = str(self.config.get("title", self.DEFAULT_TITLE))
        self._single_track_id = int(self.config.get("single_track_id", 0))
        self._track_label_prefix = str(self.config.get("track_label_prefix", "ID"))

    def analyze(self, signal: ISignal) -> TrackingPlaybackData:
        playback_frames: list[TrackingPlaybackFrame] = []
        resolved_source_path: str | None = None
        resolved_resize: tuple[int, int] | None = None
        resolved_fps: float | None = None

        for sample in signal:
            sample_metadata = dict(getattr(sample, "metadata", {}) or {})
            if resolved_source_path is None:
                source_path = sample_metadata.get("source_path")
                if source_path:
                    resolved_source_path = str(source_path)
            if resolved_resize is None:
                resolved_resize = self._normalize_resize(sample_metadata.get("resize"))
            if resolved_fps is None:
                resolved_fps = self._normalize_fps(sample_metadata.get("source_fps"))

            playback_frames.append(
                TrackingPlaybackFrame(
                    frame_index=int(sample.frame_index),
                    tracks=self._extract_tracks(sample),
                    timestamp_seconds=sample.timestamp_seconds,
                    metadata=sample_metadata,
                )
            )

        return TrackingPlaybackData(
            frames=playback_frames,
            title=self._title,
            source_path=resolved_source_path,
            resize=resolved_resize,
            fps=resolved_fps,
            metadata={
                "title": self._title,
                "sample_count": len(playback_frames),
            },
        )

    def _extract_tracks(self, sample: ISignalSample) -> list[TrackingPlaybackTrack]:
        if isinstance(sample, MultiObjectSignalSample):
            return [self._build_multi_track(track) for track in sample.tracks if track.box is not None]
        if isinstance(sample, BoxSignalSample):
            if sample.box is None:
                return []
            return [
                TrackingPlaybackTrack(
                    track_id=self._single_track_id,
                    box=sample.box,
                    centroid=sample.centroid,
                    label=f"{self._track_label_prefix} {self._single_track_id}",
                    metadata=dict(sample.metadata),
                )
            ]
        raise TypeError(
            "TrackingPlaybackAnalyzer supports only BoxSignalSample and "
            "MultiObjectSignalSample signals."
        )

    def _build_multi_track(self, track: MultiObjectTrack) -> TrackingPlaybackTrack:
        return TrackingPlaybackTrack(
            track_id=track.track_id,
            box=track.box,
            centroid=track.centroid,
            confidence=track.confidence,
            label=f"{self._track_label_prefix} {track.track_id}",
            metadata=dict(track.metadata),
        )

    @staticmethod
    def _normalize_resize(value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
        return None

    @staticmethod
    def _normalize_fps(value: Any) -> float | None:
        if value is None:
            return None
        try:
            fps = float(value)
        except (TypeError, ValueError):
            return None
        return fps if fps > 0 else None
