from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from segment_anything import sam_model_registry, SamPredictor

from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.BoxSignalSample import BoundingBox, BoxSignalSample


class SAMSingleFigureSignalExtractor(ISignalExtractor):
    """
    Segment Anything based extractor (ViT-B).
    Uses SAM to refine object mask and derive bounding box over time.
    """

    def __init__(
        self,
        start_box: BoundingBox = (0, 0, 0, 0),
        sam_checkpoint: str = "sam_vit_b_01ec64.pth",
        model_type: str = "vit_b",
        prediction_striping: int = 1,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.start_box = start_box
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type

        self._predictor: SamPredictor | None = None
        self._current_mask = None
        self.prediction_striping = prediction_striping

    # -------------------------
    # CORE PIPELINE
    # -------------------------
    def extract(self, buffer: FrameBuffer) -> ISignal:
        if self.start_box[2] <= 0 or self.start_box[3] <= 0:
            raise ValueError("start_box must have positive width and height")

        self._predictor = self._build_sam()

        samples: list[BoxSignalSample] = []
        current_box = self.start_box

        for position, frame in enumerate(buffer):
            frame_index = frame.index if frame.index is not None else position

            if position % self.prediction_striping == 0:
                image = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB)
                self._predictor.set_image(image)

                x, y, w, h = current_box
                input_box = np.array([x, y, x + w, y + h])

                masks, scores, _ = self._predictor.predict(
                    box=input_box,
                    multimask_output=False
                )
                self._current_mask = masks[0]


            # -------------------------
            # MASK -> BOX
            # -------------------------
            current_box = self._mask_to_box(self._current_mask)

            centroid = (
                current_box[0] + current_box[2] / 2.0,
                current_box[1] + current_box[3] / 2.0
            )

            # -------------------------
            # VISUALIZATION
            # -------------------------
            if self.config.get("show"):
                vis = frame.frame.copy()

                # draw mask
                vis_mask = np.zeros_like(vis)
                vis_mask[self._current_mask] = (0, 0, 255)
                vis = cv2.addWeighted(vis, 0.7, vis_mask, 0.3, 0)

                # draw box
                x, y, w, h = current_box
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # draw centroid
                cv2.circle(vis, (int(centroid[0]), int(centroid[1])), 4, (255, 0, 0), -1)

                cv2.imshow("SAM Tracking", vis)
                if cv2.waitKey(1) == 27:
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

    # -------------------------
    # SAM INIT
    # -------------------------
    def _build_sam(self) -> SamPredictor:
        BASE_DIR = Path(__file__).resolve().parent.parent
        checkpoint_path = BASE_DIR.parent / "models" / self.sam_checkpoint

        sam = sam_model_registry[self.model_type](
            checkpoint=checkpoint_path,
        )
        return SamPredictor(sam)

    # -------------------------
    # MASK -> BBOX
    # -------------------------
    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> BoundingBox:
        ys, xs = np.where(mask)

        if len(xs) == 0 or len(ys) == 0:
            return (0, 0, 0, 0)

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)