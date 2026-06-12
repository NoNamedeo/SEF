from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2

from sef.core.interfaces.ILiveAnalyzer import ILiveAnalyzer
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalExtractor import ISignalExtractor
from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.artifacts.Signal import Signal
from sef.core.artifacts.signal_sample.BoxSignalSample import BoundingBox, BoxSignalSample


class OpenCVBufferedSignalExtractor(ISignalExtractor):
    """Track a single ROI through buffered frames using an OpenCV tracker."""

    def __init__(
        self,
        tracker_type: str = "CSRT",
        start_box: BoundingBox = (0, 0, 0, 0),
        tracker_factory: Callable[[], Any] | None = None,
        live_analyzer: ILiveAnalyzer = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.tracker_type = tracker_type.upper()
        self.start_box = start_box
        self._tracker_factory = tracker_factory
        self._live_analyzer = live_analyzer

    def extract(self, buffer: FrameBuffer) -> ISignal:
        if self.start_box[2] <= 0 or self.start_box[3] <= 0:
            raise ValueError("example_box must have positive width and height")

        tracker = self._build_tracker()
        samples: list[BoxSignalSample] = []
        current_box: BoundingBox | None = None

        for position, frame in enumerate(buffer):
            frame_index = frame.index if frame.index is not None else position

            if position == 0:
                tracker.init(frame.frame, self.start_box)
                current_box = self.start_box
            else:
                success, updated_box = tracker.update(frame.frame)
                current_box = self._normalize_box(updated_box) if success else None

            centroid = None
            if current_box is not None:
                x, y, w, h = current_box
                centroid = (x + w / 2.0, y + h / 2.0)



            if self._live_analyzer is not None and self.config.get("show_graph"):
                self.update(frame_index, current_box, centroid, frame)


            if self.config.get("show_contours") and current_box is not None:
                x, y, w, h = current_box
                roi = frame.frame[y:y + h, x:x + w]

                if roi.size != 0:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (5, 5), 0)

                    _, thresh = cv2.threshold(
                        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    contours, _ = cv2.findContours(
                        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for cnt in contours:
                        cnt[:, 0, 0] += x
                        cnt[:, 0, 1] += y
                        cv2.drawContours(frame.frame, [cnt], -1, (0, 0, 255), 2)

            if self.config.get("show"):
                if current_box is not None:
                    cv2.rectangle(frame.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow("Tracking", frame.frame)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC
                        break

            samples.append(
                BoxSignalSample(
                    frame_index=frame_index,
                    box=current_box,
                    centroid=centroid,
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata=dict(frame.metadata),
                )
            )

        return Signal(samples)

    def track(self, buffer: FrameBuffer) -> ISignal:
        """Backward-compatible alias."""
        return self.extract(buffer)

    def _build_tracker(self):
        if self._tracker_factory is not None:
            return self._tracker_factory()
        return self._create_tracker(self.tracker_type)

    @staticmethod
    def _normalize_box(box) -> BoundingBox:
        x, y, w, h = box
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _create_tracker(tracker_type: str):
        tracker_factories = {
            "CSRT": [("legacy", "TrackerCSRT_create"), (None, "TrackerCSRT_create")],
            "KCF": [("legacy", "TrackerKCF_create"), (None, "TrackerKCF_create")],
            "MIL": [("legacy", "TrackerMIL_create"), (None, "TrackerMIL_create")],
            "GOTURN": [(None, "TrackerGOTURN_create")],
        }

        if tracker_type not in tracker_factories:
            raise ValueError(f"Tracker {tracker_type} non supportato")

        for namespace, factory_name in tracker_factories[tracker_type]:
            module = cv2 if namespace is None else getattr(cv2, namespace, None)
            if module is None:
                continue

            factory = getattr(module, factory_name, None)
            if factory is not None:
                return factory()

        raise ValueError(f"Tracker factory not available for {tracker_type}")

    def update(self, frame_index, current_box, centroid, frame):
        self._live_analyzer.update(BoxSignalSample(
            frame_index=frame_index,
            box=current_box,
            centroid=centroid,
            timestamp_seconds=frame.timestamp_seconds,
            metadata=dict(frame.metadata),
        ))