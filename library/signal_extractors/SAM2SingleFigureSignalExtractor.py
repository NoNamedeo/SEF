from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.BoxSignalSample import BoundingBox, BoxSignalSample


class SAM2SingleFigureSignalExtractor(ISignalExtractor):
    """
    Track a single object using SAM2 video segmentation.
    Input: coarse bounding box
    Output: per-frame mask -> centroid -> synthetic bounding box
    """

    def __init__(
        self,
        start_box: BoundingBox = (0, 0, 0, 0),
        sam2_predictor_factory: Callable[[], Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.start_box = start_box
        self._sam2_factory = sam2_predictor_factory

        self._predictor = None
        self._state = None

    def extract(self, buffer: FrameBuffer) -> ISignal:
        x, y, w, h = self.start_box
        if w <= 0 or h <= 0:
            raise ValueError("start_box must have positive width and height")

        samples: list[BoxSignalSample] = []

        self._predictor = self._build_sam2()

        # --- SAM2 initialization ---
        first_frame = buffer[0].frame

        # initialize video state
        self._predictor.init_state(first_frame)

        # register initial box prompt
        self._predictor.add_new_prompt(
            frame_idx=0,
            obj_id=1,
            box=np.array([x, y, x + w, y + h]),
        )

        for position, frame in enumerate(buffer):
            frame_index = frame.index if frame.index is not None else position

            # --- propagate mask for current frame ---
            masks = self._predictor.propagate(frame.frame)

            # assume single object (obj_id = 1)
            mask = masks.get(1, None)

            centroid = None
            box = None

            if mask is not None:
                ys, xs = np.where(mask > 0)

                if len(xs) > 0 and len(ys) > 0:
                    cx = float(xs.mean())
                    cy = float(ys.mean())
                    centroid = (cx, cy)

                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()

                    box = (
                        int(x_min),
                        int(y_min),
                        int(x_max - x_min),
                        int(y_max - y_min),
                    )

            if self.config.get("show"):
                import cv2

                display = frame.frame.copy()

                if mask is not None:
                    colored = np.zeros_like(display)
                    colored[:, :, 1] = mask * 255
                    display = cv2.addWeighted(display, 1.0, colored, 0.5, 0)

                if box is not None:
                    x, y, w, h = box
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if centroid is not None:
                    cv2.circle(
                        display,
                        (int(centroid[0]), int(centroid[1])),
                        4,
                        (0, 0, 255),
                        -1,
                    )

                cv2.imshow("SAM2 Tracking", display)
                key = cv2.waitKey(1)
                if key == 27:
                    break

            samples.append(
                BoxSignalSample(
                    frame_index=frame_index,
                    box=box,
                    centroid=centroid,
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata=dict(frame.metadata),
                )
            )

        return Signal(samples)

    def track(self, buffer: FrameBuffer) -> ISignal:
        return self.extract(buffer)

    def _build_sam2(self):
        if self._sam2_factory is not None:
            return self._sam2_factory()
        return self._default_sam2_factory()

    def _default_sam2_factory(self):
        """
        Default SAM2 loader.
        Lazy import to avoid heavy startup cost.
        """

        import os

        from external.sam2.sam2.build_sam import build_sam2_video_predictor
        from external.sam2.sam2.sam2_video_predictor import SAM2VideoPredictor

        # root del package SAM2 (external/sam2)
        SAM2_ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        # checkpoint assoluto
        checkpoint = self.config.get(
            "sam2_checkpoint",
            os.path.join(
                SAM2_ROOT,
                "checkpoints",
                "sam2_hiera_tiny.pt"
            )
        )

        # config assoluta (Hydra-safe)
        model_cfg = self.config.get(
            "sam2_config",
            os.path.join(
                SAM2_ROOT,
                "sam2",
                "configs",
                "sam2",
                "sam2_hiera_l.yaml"
            )
        )

        predictor: SAM2VideoPredictor = build_sam2_video_predictor(
            model_cfg,
            checkpoint
        )

        return predictor
