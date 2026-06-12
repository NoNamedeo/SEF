from __future__ import annotations

from collections.abc import Callable
from typing import Any, List

import cv2

from sef.core.artifacts.signal_sample.MultiManualSignalSample import MultiManualSignalSample
from sef.core.interfaces.ILiveAnalyzer import ILiveAnalyzer
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalExtractor import ISignalExtractor
from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.artifacts.Signal import Signal
from sef.core.artifacts.signal_sample.BoxSignalSample import BoundingBox, BoxSignalSample


class OpenCVMultiManualSignalExtractor(ISignalExtractor):
    """Track multiple ROIs through buffered frames using independent OpenCV trackers."""

    def __init__(
        self,
        tracker_type: str = "CSRT",
        start_boxes: List[BoundingBox] | None = None,
        tracker_factory: Callable[[], Any] | None = None,
        live_analyzer: ILiveAnalyzer = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config or {})
        self.tracker_type = tracker_type.upper()
        self.start_boxes = start_boxes or []
        self._tracker_factory = tracker_factory
        self._live_analyzer = live_analyzer

        self.object_ids = list(range(len(self.start_boxes)))

    def extract(self, buffer: FrameBuffer) -> ISignal:
        if not self.start_boxes:
            raise ValueError("start_boxes must contain at least one bounding box")

        for box in self.start_boxes:
            if box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"Invalid box: {box}")

        trackers = [self._build_tracker() for _ in self.start_boxes]
        current_boxes: List[BoundingBox | None] = list(self.start_boxes)

        samples: list[MultiManualSignalSample] = []

        for position, frame in enumerate(buffer):
            frame_index = frame.index if frame.index is not None else position

            if position == 0:
                for tracker, box in zip(trackers, self.start_boxes):
                    tracker.init(frame.frame, box)

            else:
                for i in range(len(trackers)):
                    success, updated_box = trackers[i].update(frame.frame)

                    if success:
                        current_boxes[i] = self._normalize_box(updated_box)
                    else:
                        current_boxes[i] = None

            if self.config.get("show"):
                for i, box in enumerate(current_boxes):
                    if box is None:
                        continue

                    x, y, w, h = box
                    obj_id = self.object_ids[i]

                    cv2.rectangle(
                        frame.frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame.frame,
                        f"ID {obj_id}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Multi Tracking", frame.frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break

            sample = MultiManualSignalSample(samples={})

            for i, box in enumerate(current_boxes):
                obj_id = self.object_ids[i]

                if box is None:
                    continue

                x, y, w, h = box
                centroid = (x + w / 2.0, y + h / 2.0)

                sample.samples[obj_id] = BoxSignalSample(
                    frame_index=frame_index,
                    box=box,
                    centroid=centroid,
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata={
                        **frame.metadata,
                        "object_id": obj_id,
                    },
                )

            samples.append(sample)

        return Signal(samples)

    def track(self, buffer: FrameBuffer) -> ISignal:
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