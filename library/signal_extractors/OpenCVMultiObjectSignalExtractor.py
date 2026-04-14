from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.MultiObjectSignalSample import (
    BoundingBox,
    MultiObjectSignalSample,
    MultiObjectTrack,
)
from library.core.artifacts.Signal import Signal
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor


class OpenCVMultiObjectSignalExtractor(ISignalExtractor):
    """
    Multi-object tracker starting from a seed ROI.
    Expands tracking to similar objects in the scene.
    """

    def __init__(
        self,
        tracker_type: str = "CSRT",
        start_box: BoundingBox = (0, 0, 0, 0),
        max_objects: int = 10,
        similarity_threshold: float = 0.6,
        tracker_factory: Callable[[], Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

        self.tracker_type = tracker_type.upper()
        self.start_box = start_box
        self.max_objects = max_objects
        self.similarity_threshold = similarity_threshold
        self._tracker_factory = tracker_factory

        self._trackers: dict[int, Any] = {}
        self._next_id = 0

    def extract(self, buffer: FrameBuffer) -> ISignal:
        if self.start_box[2] <= 0 or self.start_box[3] <= 0:
            raise ValueError("example_box must have positive width and height")

        samples: list[MultiObjectSignalSample] = []

        show = bool(self.config.get("show", False))

        for position, frame in enumerate(buffer):
            frame_index = frame.index if frame.index is not None else position
            img = frame.frame

            # caso del primo frame
            if position == 0:
                self._init_seed_tracker(img)

            tracks = self._update_all_trackers(img)

            # aggiungo nuovi oggetti da tracciare, se ne trovo altri
            if len(self._trackers) < self.max_objects:
                self._expand_tracks(img, tracks)

            if show:
                # visualizzo cosa sta succedendo

                vis_frame = img.copy()

                for t in tracks:
                    if t.box is None:
                        continue

                    x, y, w, h = t.box
                    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.putText(vis_frame, f"ID {t.track_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("MultiObject Tracking", vis_frame)

                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break

            sample = MultiObjectSignalSample(
                frame_index=frame_index,
                tracks=tracks,
                timestamp_seconds=frame.timestamp_seconds,
                metadata=dict(frame.metadata),
            )

            samples.append(sample)

        cv2.destroyAllWindows()

        return Signal(samples)

    def _init_seed_tracker(self, frame: np.ndarray):
        tracker = self._build_tracker()
        tracker.init(frame, self.start_box)

        self._trackers[self._next_id] = tracker
        self._next_id += 1

    def _update_all_trackers(self, frame: np.ndarray) -> list[MultiObjectTrack]:
        results: list[MultiObjectTrack] = []

        dead_ids = []

        for track_id, tracker in self._trackers.items():
            success, box = tracker.update(frame)

            if not success:
                dead_ids.append(track_id)
                continue

            x, y, w, h = self._normalize_box(box)
            centroid = (x + w / 2.0, y + h / 2.0)

            results.append(
                MultiObjectTrack(
                    track_id=track_id,
                    box=(x, y, w, h),
                    centroid=centroid,
                )
            )

        # remove lost trackers
        for tid in dead_ids:
            del self._trackers[tid]

        return results

    def _expand_tracks(self, frame: np.ndarray, existing_tracks: list[MultiObjectTrack]):
        """
        Find new objects similar to the seed and start tracking them.
        Default heuristic: simple patch similarity (HSV histogram).
        TODO: questo è detection, non tracking, si potrebbe pensare di usare classe/i a parte
        """

        # istogramma di partenza (detecterà in base a questo)
        seed_hist = self._compute_reference_hist(frame, self.start_box)

        h, w = frame.shape[:2]
        step = 40  # sliding window coarse search

        for y in range(0, h - step, step):
            for x in range(0, w - step, step):
                candidate_box = (x, y, step, step)

                # controllo se sto sopra a una casella gia tracciata
                if self._overlaps_existing(candidate_box, existing_tracks):
                    continue

                candidate_hist = self._compute_reference_hist(frame, candidate_box)
                score = cv2.compareHist(seed_hist, candidate_hist, cv2.HISTCMP_CORREL)

                if score > self.similarity_threshold:
                    self._add_tracker(frame, candidate_box)

                if len(self._trackers) >= self.max_objects:
                    return

    def _add_tracker(self, frame: np.ndarray, box: BoundingBox):
        tracker = self._build_tracker()
        tracker.init(frame, box)

        self._trackers[self._next_id] = tracker
        self._next_id += 1

    @staticmethod
    def _compute_reference_hist(frame: np.ndarray, box: BoundingBox):
        x, y, w, h = box
        # seleziono la box tracciata
        roi = frame[y : y + h, x : x + w]
        # la trasformo in hsv
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # usa i canali 0 e 1 (H e S), crea un istogramma 2d da 30 bin per hue e 32 bin
        # per saturation (matrice 30x32)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    @staticmethod
    def _overlaps_existing(box: BoundingBox, tracks: list[MultiObjectTrack]) -> bool:
        x, y, w, h = box

        for t in tracks:
            if t.box is None:
                continue
            tx, ty, tw, th = t.box

            if x < tx + tw and x + w > tx and y < ty + th and y + h > ty:
                # TODO: si potrebbe cambiare questo if costringendo a non avere sovrapposizione per nulla
                # (e non come adesso che ammette sovrapposizione solo se totale)
                return True

        return False

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
