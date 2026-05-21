from __future__ import annotations

import urllib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.interfaces.StreamingContracts import IStreamingSignalExtractor
from library.core.interfaces.ILiveAnalyzer import ILiveAnalyzer

from library.core.artifacts.COCOSkeletonSignalSample import COCOSkeletonSignalSample


class YOLOSkeletonCOCOStreamSignalExtractor(IStreamingSignalExtractor):
    """
    Streaming COCO pose extractor using YOLO (Ultralytics).
    Produces COCOSkeletonSignalSample per frame.
    """

    capabilities = StageCapabilities.streaming(
        stateful=False,
        preserves_order=True,
        realtime_safe=True,
    )

    #lista ordinata dei 17 punti che vado a prendere nello skeleton
    COCO_17 = [
        "nose",
        "left_eye", "right_eye",
        "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]

    #per disegnare lo stickman
    COCO_EDGES = [
        (5, 6),  # shoulders
        (5, 7), (7, 9),  # left arm
        (6, 8), (8, 10),  # right arm
        (5, 11), (6, 12),  # torso
        (11, 12),  # hips
        (11, 13), (13, 15),  # left leg
        (12, 14), (14, 16)  # right leg
    ]

    def __init__(
        self,
        model_name: str = "yolo11s-pose.pt", #c'è anche yolov8n-pose.pt (più veloce)
        buffer: SignalBuffer | None = None,
        live_analyzer: ILiveAnalyzer | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self._model = self.load_model(model_name)
        self._default_buffer = buffer
        self._live_analyzer = live_analyzer

    def extract(self, buffer: FrameBuffer) -> SignalBuffer:
        output = self._default_buffer or SignalBuffer()
        self.extract_into(buffer, output)
        return output

    def extract_into(self, buffer: FrameBuffer, output_buffer: SignalBuffer) -> None:
        try:
            for position, frame in enumerate(buffer):
                frame_index = frame.index if frame.index is not None else position

                result = self._model(frame.frame, verbose=False)[0]

                skeleton, conf = self._parse_result(result)

                centroid = self._compute_centroid(skeleton)

                sample = COCOSkeletonSignalSample(
                    frame_index=frame_index,
                    skeleton=skeleton,
                    confidence=conf,
                    centroid=centroid,
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata={
                        "source": "yolo11-pose",
                    },
                )

                if self.config.get("show"):
                    vis_frame = frame.frame.copy()

                    self._draw_skeleton(vis_frame, skeleton, conf)

                    cv2.circle(
                        vis_frame,
                        (int(centroid[0]), int(centroid[1])),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    cv2.imshow("YOLO Pose COCO", vis_frame)

                    key = cv2.waitKey(1)
                    if key == 27:  # ESC
                        break

                if self._live_analyzer is not None and self.config.get("show_graph"):
                    self.update(sample)

                output_buffer.put(sample)

        finally:
            output_buffer.close()

    def _parse_result(self, result):
        """
        Returns:
            skeleton: [17,2]
            conf: [17]
        """
        if result.keypoints is None:
            return np.zeros((17, 2)), np.zeros(17)

        kpts = result.keypoints

        xy = kpts.xy[0].cpu().numpy()        # [17,2]
        conf = kpts.conf[0].cpu().numpy()    # [17]

        return xy, conf

    def _compute_centroid(self, skeleton):
        """
        Stable centroid from hips
        """
        left_hip = skeleton[11]
        right_hip = skeleton[12]

        return (
            (left_hip[0] + right_hip[0]) / 2.0,
            (left_hip[1] + right_hip[1]) / 2.0,
        )

    def _draw_skeleton(self, frame_img, skeleton, conf=None, threshold=0.3):
        """
        Draw COCO skeleton on frame
        """

        # draw joints
        for i, (x, y) in enumerate(skeleton):
            if conf is not None and conf[i] < threshold:
                continue

            cv2.circle(
                frame_img,
                (int(x), int(y)),
                4,
                (0, 255, 0),
                -1
            )

        # draw edges
        for a, b in self.COCO_EDGES:
            if conf is not None and (conf[a] < threshold or conf[b] < threshold):
                continue

            pt1 = tuple(map(int, skeleton[a]))
            pt2 = tuple(map(int, skeleton[b]))

            cv2.line(frame_img, pt1, pt2, (255, 0, 0), 2)

    def load_model(self, model_name: str = "yolo11-pose.pt") -> YOLO:
        base_dir = Path(__file__).resolve().parents[1]
        model_dir = base_dir / "YOLOPoseModels"
        model_dir.mkdir(exist_ok=True, parents=True)

        model_path = model_dir / model_name

        # se non esiste, lo scarico (NOTA: avevo gia installato il modello, quindi non so se il link funzioni)
        if not model_path.exists():
            url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt"
            urllib.request.urlretrieve(url, model_path)

        return YOLO(str(model_path))

    def update(self, sample: COCOSkeletonSignalSample):
        if self._live_analyzer is None:
            return

        self._live_analyzer.update(sample)

    def track(self, buffer: FrameBuffer) -> SignalBuffer:
        return self.extract(buffer)