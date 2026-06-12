from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.artifacts.Signal import Signal
from sef.core.artifacts.signal_sample.MultiObjectSignalSample import (
    BoundingBox,
    MultiObjectSignalSample,
    MultiObjectTrack,
)
from sef.core.interfaces.IEventEmitter import IEventEmitter
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalExtractor import ISignalExtractor

log = logging.getLogger(__name__)


class OpenCVMultiObjectSignalExtractor(ISignalExtractor, IEventEmitter):
    """
    Multi-object tracker initialized from a manually selected seed ROI.

    Intended workflow
    -----------------
    1. The caller selects one seed ROI manually (for example one cross marker).
    2. On the first frame, the extractor initializes the seed tracker.
    3. It optionally detects additional similar objects by template matching.
    4. It tracks all detected objects across subsequent frames.

    Notes
    -----
    - This extractor is best suited for scenarios with repeated visual patterns
      (for example multiple crosses with similar appearance).
    - For micromovement analysis, the tracked bounding box centroid may later be
      refined with a subpixel center estimator in a dedicated analyzer/cleaner.
    """

    def __init__(
        self,
        tracker_type: str = "CSRT",
        start_box: BoundingBox | None = None,
        roi_initializer: str = "manual_seed_template_expand",
        max_objects: int = 3,
        template_match_threshold: float = 0.88,
        similarity_threshold: float | None = None,
        min_detection_distance: int = 30,
        auto_detect_on_first_frame: bool = True,
        re_detect_lost_objects: bool = False,
        tracker_factory: Callable[[], Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

        self.tracker_type = tracker_type.upper()
        self.start_box = start_box
        self.roi_initializer = roi_initializer
        self.max_objects = max_objects
        if similarity_threshold is not None:
            template_match_threshold = similarity_threshold
        self.template_match_threshold = template_match_threshold
        self.min_detection_distance = min_detection_distance
        self.auto_detect_on_first_frame = auto_detect_on_first_frame
        self.re_detect_lost_objects = re_detect_lost_objects
        self._tracker_factory = tracker_factory

        self._trackers: dict[int, Any] = {}
        self._next_id = 0
        self._seed_template_gray: np.ndarray | None = None
        self._seed_box_initialized = False
        self._preview_disabled = False

    def extract(self, buffer: FrameBuffer) -> ISignal:
        self._validate_configuration()
        self._reset_runtime_state()

        samples: list[MultiObjectSignalSample] = []
        show = bool(self.config.get("show", False))

        for position, frame in enumerate(buffer):
            frame_index = frame.index if frame.index is not None else position
            image = frame.frame

            if position == 0:
                self._bootstrap_trackers(image, frame_index)

            tracks, lost_track_ids = self._update_all_trackers(image, frame_index=frame_index)

            if self.re_detect_lost_objects and lost_track_ids and len(self._trackers) < self.max_objects:
                self._expand_tracks_from_template(
                    frame=image,
                    existing_tracks=tracks,
                    frame_index=frame_index,
                )
                tracks, _ = self._update_all_trackers(image, frame_index=frame_index)

            if show and not self._preview_disabled:
                try:
                    self._show_tracking_debug_window(image=image, tracks=tracks)
                except cv2.error as exc:
                    self._preview_disabled = True
                    log.warning(
                        "OpenCV tracking preview unavailable; continuing without a live window: %s",
                        exc,
                    )

            sample = MultiObjectSignalSample(
                frame_index=frame_index,
                tracks=tracks,
                timestamp_seconds=frame.timestamp_seconds,
                metadata=dict(frame.metadata),
            )
            samples.append(sample)

        self._close_tracking_windows()
        return Signal(samples)

    def _validate_configuration(self) -> None:
        if self.start_box is None:
            raise ValueError("start_box is required and must be provided by the caller")

        x, y, w, h = self.start_box
        if w <= 0 or h <= 0:
            raise ValueError("start_box must have positive width and height")

        if self.max_objects <= 0:
            raise ValueError("max_objects must be greater than zero")

        if not 0.0 <= self.template_match_threshold <= 1.0:
            raise ValueError("template_match_threshold must be between 0.0 and 1.0")

        if self.min_detection_distance < 0:
            raise ValueError("min_detection_distance must be non-negative")

    def _reset_runtime_state(self) -> None:
        self._trackers.clear()
        self._next_id = 0
        self._seed_template_gray = None
        self._seed_box_initialized = False

    def _bootstrap_trackers(self, frame: np.ndarray, frame_index: int) -> None:
        seed_box = self._clamp_box_to_frame(self.start_box, frame)
        self.start_box = seed_box
        self._seed_template_gray = self._extract_template_gray(frame, seed_box)

        self._init_seed_tracker(frame, seed_box)

        if self.roi_initializer == "manual_seed_template_expand" and self.auto_detect_on_first_frame and self.max_objects > 1:
            seed_track = self._build_track_info(track_id=0, box=seed_box)
            self._expand_tracks_from_template(
                frame=frame,
                existing_tracks=[seed_track],
                frame_index=frame_index,
            )

        self._seed_box_initialized = True

    def _init_seed_tracker(self, frame: np.ndarray, seed_box: BoundingBox) -> None:
        tracker = self._build_tracker()
        tracker.init(frame, seed_box)

        track_id = self._next_id
        self._trackers[track_id] = tracker
        self._next_id += 1

        self.emit(
            "track_created",
            {
                "track_id": track_id,
                "box": seed_box,
                "kind": "seed",
                "source_path": self.config.get("source_path"),
            },
        )

    def _update_all_trackers(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> tuple[list[MultiObjectTrack], list[int]]:
        results: list[MultiObjectTrack] = []
        dead_ids: list[int] = []

        for track_id, tracker in list(self._trackers.items()):
            success, raw_box = tracker.update(frame)

            if not success:
                dead_ids.append(track_id)
                continue

            normalized_box = self._normalize_box(raw_box)
            clamped_box = self._clamp_box_to_frame(normalized_box, frame)

            if not self._is_valid_box(clamped_box):
                dead_ids.append(track_id)
                continue

            results.append(self._build_track_info(track_id=track_id, box=clamped_box))

        for track_id in dead_ids:
            self._trackers.pop(track_id, None)
            self.emit(
                "track_lost",
                {
                    "track_id": track_id,
                    "frame_index": frame_index,
                    "source_path": self.config.get("source_path"),
                },
            )

        return results, dead_ids

    def _expand_tracks_from_template(
        self,
        frame: np.ndarray,
        existing_tracks: list[MultiObjectTrack],
        frame_index: int,
    ) -> None:
        if self._seed_template_gray is None:
            return

        if len(self._trackers) >= self.max_objects:
            return

        candidate_boxes = self._detect_similar_objects_by_template(
            frame=frame,
            existing_tracks=existing_tracks,
        )

        for candidate_box in candidate_boxes:
            if len(self._trackers) >= self.max_objects:
                return
            self._add_tracker(frame=frame, box=candidate_box, frame_index=frame_index)

    def _detect_similar_objects_by_template(
        self,
        frame: np.ndarray,
        existing_tracks: list[MultiObjectTrack],
    ) -> list[BoundingBox]:
        if self._seed_template_gray is None:
            return []

        frame_gray = self._to_gray(frame)
        template_gray = self._seed_template_gray

        template_height, template_width = template_gray.shape[:2]
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        ys, xs = np.where(result >= self.template_match_threshold)

        scored_candidates: list[tuple[float, BoundingBox]] = []
        for y, x in zip(ys, xs):
            score = float(result[y, x])
            box = (int(x), int(y), int(template_width), int(template_height))
            scored_candidates.append((score, box))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)

        selected_boxes: list[BoundingBox] = []

        for _, candidate_box in scored_candidates:
            candidate_box = self._clamp_box_to_frame(candidate_box, frame)

            if not self._is_valid_box(candidate_box):
                continue

            if self._is_near_box(candidate_box, self.start_box, self.min_detection_distance):
                continue

            if self._overlaps_existing(candidate_box, existing_tracks):
                continue

            if any(self._is_near_box(candidate_box, selected_box, self.min_detection_distance) for selected_box in selected_boxes):
                continue

            selected_boxes.append(candidate_box)

            current_total = len(existing_tracks) + len(selected_boxes)
            if current_total >= self.max_objects:
                break

        return selected_boxes

    def _add_tracker(self, frame: np.ndarray, box: BoundingBox, frame_index: int) -> None:
        tracker = self._build_tracker()
        tracker.init(frame, box)

        track_id = self._next_id
        self._trackers[track_id] = tracker
        self._next_id += 1

        self.emit(
            "track_created",
            {
                "track_id": track_id,
                "box": box,
                "frame_index": frame_index,
                "kind": "expanded",
                "source_path": self.config.get("source_path"),
            },
        )

    def _build_track_info(self, track_id: int, box: BoundingBox) -> MultiObjectTrack:
        x, y, w, h = box
        centroid = (x + w / 2.0, y + h / 2.0)

        return MultiObjectTrack(
            track_id=track_id,
            box=box,
            centroid=centroid,
        )

    def _show_tracking_debug_window(
        self,
        image: np.ndarray,
        tracks: list[MultiObjectTrack],
    ) -> None:
        vis_frame = image.copy()

        for track in tracks:
            if track.box is None:
                continue

            x, y, w, h = track.box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis_frame,
                f"ID {track.track_id}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            cx, cy = track.centroid
            cv2.circle(vis_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        cv2.imshow("MultiObject Tracking", vis_frame)

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Tracking interrupted by user")

    @staticmethod
    def _close_tracking_windows() -> None:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # HighGUI is not available in headless environments. The pipeline
            # should still complete and return the tracking results.
            return

    def _build_tracker(self):
        if self._tracker_factory is not None:
            return self._tracker_factory()
        return self._create_tracker(self.tracker_type)

    def _extract_template_gray(self, frame: np.ndarray, box: BoundingBox) -> np.ndarray:
        x, y, w, h = self._clamp_box_to_frame(box, frame)
        roi = frame[y : y + h, x : x + w]

        if roi.size == 0:
            raise ValueError("The selected start_box produced an empty ROI")

        return self._to_gray(roi)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _normalize_box(box: Any) -> BoundingBox:
        x, y, w, h = box
        return int(round(x)), int(round(y)), int(round(w)), int(round(h))

    @staticmethod
    def _is_valid_box(box: BoundingBox) -> bool:
        _, _, w, h = box
        return w > 0 and h > 0

    @staticmethod
    def _clamp_box_to_frame(box: BoundingBox, frame: np.ndarray) -> BoundingBox:
        frame_height, frame_width = frame.shape[:2]
        x, y, w, h = box

        x = max(0, min(int(x), frame_width - 1))
        y = max(0, min(int(y), frame_height - 1))
        w = max(1, min(int(w), frame_width - x))
        h = max(1, min(int(h), frame_height - y))

        return x, y, w, h

    @staticmethod
    def _overlaps_existing(box: BoundingBox, tracks: list[MultiObjectTrack]) -> bool:
        x, y, w, h = box

        for track in tracks:
            if track.box is None:
                continue

            tx, ty, tw, th = track.box

            if x < tx + tw and x + w > tx and y < ty + th and y + h > ty:
                return True

        return False

    @staticmethod
    def _is_near_box(
        first_box: BoundingBox,
        second_box: BoundingBox | None,
        min_distance: int,
    ) -> bool:
        if second_box is None:
            return False

        ax, ay, aw, ah = first_box
        bx, by, bw, bh = second_box

        first_center_x = ax + aw / 2.0
        first_center_y = ay + ah / 2.0
        second_center_x = bx + bw / 2.0
        second_center_y = by + bh / 2.0

        distance = ((first_center_x - second_center_x) ** 2 + (first_center_y - second_center_y) ** 2) ** 0.5
        return distance < min_distance

    @staticmethod
    def _create_tracker(tracker_type: str):
        tracker_factories = {
            "CSRT": [("legacy", "TrackerCSRT_create"), (None, "TrackerCSRT_create")],
            "KCF": [("legacy", "TrackerKCF_create"), (None, "TrackerKCF_create")],
            "MIL": [("legacy", "TrackerMIL_create"), (None, "TrackerMIL_create")],
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
