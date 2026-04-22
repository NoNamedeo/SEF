from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from library.core.artifacts.ArucoMarkerSignalSample import (
    ArucoMarkerObservation,
    ArucoMarkerSignalSample,
    MarkerCorners,
)
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor


class ArucoMarkerSignalExtractor(ISignalExtractor):
    """Detect DICT_6X6_250 ArUco markers frame by frame."""

    DICTIONARY_ID = cv2.aruco.DICT_6X6_250
    DEFAULT_WHITE_BORDER_PADDING_PX = 32

    def __init__(
        self,
        marker_ids: Sequence[int] | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.marker_ids = tuple(sorted({int(marker_id) for marker_id in marker_ids})) if marker_ids is not None else None
        self._white_border_padding_px = int(
            self.config.get("white_border_padding_px", self.DEFAULT_WHITE_BORDER_PADDING_PX)
        )
        if self._white_border_padding_px < 0:
            raise ValueError("white_border_padding_px must be non-negative")
        self._dictionary = cv2.aruco.getPredefinedDictionary(self.DICTIONARY_ID)
        self._detector_parameters = self._build_detector_parameters()
        self._detector = self._build_detector()

    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples: list[ArucoMarkerSignalSample] = []
        known_marker_ids = set(self.marker_ids or ())

        for position, frame in enumerate(buffer):
            frame_index = frame.index if frame.index is not None else position
            grayscale = self._to_grayscale(frame.frame)
            corners, ids, rejected = self._detect_markers(grayscale)

            detected_observations = self._build_detected_observations(
                corners=corners,
                ids=ids,
                frame_shape=grayscale.shape,
            )

            known_marker_ids.update(detected_observations.keys())
            observations = self._build_frame_observations(
                detected_observations=detected_observations,
                known_marker_ids=known_marker_ids,
            )

            samples.append(
                ArucoMarkerSignalSample(
                    frame_index=int(frame_index),
                    markers=observations,
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata={
                        **dict(frame.metadata),
                        "detected_marker_count": sum(1 for observation in observations if observation.detected),
                        "rejected_candidate_count": len(rejected),
                    },
                )
            )

        return Signal(samples)

    def _build_frame_observations(
        self,
        *,
        detected_observations: dict[int, ArucoMarkerObservation],
        known_marker_ids: set[int],
    ) -> list[ArucoMarkerObservation]:
        if self.marker_ids is not None:
            frame_marker_ids = self.marker_ids
        elif known_marker_ids:
            frame_marker_ids = tuple(sorted(known_marker_ids))
        else:
            frame_marker_ids = tuple(sorted(detected_observations))

        observations: list[ArucoMarkerObservation] = []
        for marker_id in frame_marker_ids:
            observation = detected_observations.get(marker_id)
            if observation is None:
                observations.append(
                    ArucoMarkerObservation(
                        marker_id=marker_id,
                        corners=None,
                        center_x=None,
                        center_y=None,
                        detected=False,
                        quality_score=None,
                        metadata={"reason": "not_detected"},
                    )
                )
                continue
            observations.append(observation)
        return observations

    def _build_detected_observations(
        self,
        *,
        corners: list[np.ndarray],
        ids: np.ndarray | None,
        frame_shape: tuple[int, int],
    ) -> dict[int, ArucoMarkerObservation]:
        if ids is None or len(corners) == 0:
            return {}

        observations: dict[int, ArucoMarkerObservation] = {}
        frame_height, frame_width = frame_shape[:2]
        frame_area = float(frame_height * frame_width) if frame_height > 0 and frame_width > 0 else 0.0

        for marker_id, raw_corners in zip(ids.flatten().tolist(), corners):
            normalized_corners = self._normalize_corners(raw_corners)
            area_px = float(abs(cv2.contourArea(np.asarray(normalized_corners, dtype=np.float32))))
            center_x = sum(point[0] for point in normalized_corners) / 4.0
            center_y = sum(point[1] for point in normalized_corners) / 4.0
            observations[int(marker_id)] = ArucoMarkerObservation(
                marker_id=int(marker_id),
                corners=normalized_corners,
                center_x=center_x,
                center_y=center_y,
                detected=True,
                quality_score=(area_px / frame_area) if frame_area > 0 else None,
                metadata={"area_px": area_px},
            )
        return observations

    def _detect_markers(
        self,
        grayscale_frame: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
        corners, ids, rejected = self._detect_markers_raw(grayscale_frame)
        if ids is not None and len(corners) > 0:
            return list(corners or []), ids, list(rejected or [])

        if self._white_border_padding_px == 0:
            return list(corners or []), ids, list(rejected or [])

        padded_frame = cv2.copyMakeBorder(
            grayscale_frame,
            self._white_border_padding_px,
            self._white_border_padding_px,
            self._white_border_padding_px,
            self._white_border_padding_px,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        padded_corners, padded_ids, padded_rejected = self._detect_markers_raw(padded_frame)
        if padded_ids is None or len(padded_corners) == 0:
            return list(corners or []), ids, list(rejected or padded_rejected or [])

        adjusted_corners = [
            np.asarray(raw_corners, dtype=np.float32) - np.array(
                [[[self._white_border_padding_px, self._white_border_padding_px]]],
                dtype=np.float32,
            )
            for raw_corners in padded_corners
        ]
        return list(adjusted_corners), padded_ids, list(padded_rejected or [])

    def _detect_markers_raw(
        self,
        grayscale_frame: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
        if self._detector is not None:
            corners, ids, rejected = self._detector.detectMarkers(grayscale_frame)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                grayscale_frame,
                self._dictionary,
                parameters=self._detector_parameters,
            )
        return list(corners or []), ids, list(rejected or [])

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 1:
            return frame[:, :, 0]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _normalize_corners(raw_corners: np.ndarray) -> MarkerCorners:
        squeezed = np.asarray(raw_corners, dtype=np.float32).reshape(4, 2)
        return tuple((float(x), float(y)) for x, y in squeezed)  # type: ignore[return-value]

    @staticmethod
    def _build_detector_parameters():
        if hasattr(cv2.aruco, "DetectorParameters"):
            return cv2.aruco.DetectorParameters()
        return cv2.aruco.DetectorParameters_create()

    def _build_detector(self):
        if hasattr(cv2.aruco, "ArucoDetector"):
            return cv2.aruco.ArucoDetector(self._dictionary, self._detector_parameters)
        return None
