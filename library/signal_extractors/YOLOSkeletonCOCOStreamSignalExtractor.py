from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from library.core.artifacts.COCOSkeletonSignalSample import COCOSkeletonSignalSample
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.interfaces.ILiveAnalyzer import ILiveAnalyzer
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.interfaces.StreamingContracts import IStreamingSignalExtractor

if TYPE_CHECKING:
    from ultralytics import YOLO


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

    # lista ordinata dei 17 punti che vado a prendere nello skeleton
    COCO_17 = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(
        self,
        model_name: str = "yolo11s-pose.pt",  # c'è anche yolov8n-pose.pt (più veloce)
        buffer: SignalBuffer | None = None,
        live_analyzer: ILiveAnalyzer | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self._model = self.load_model(model_name)
        self._default_buffer = buffer
        self._live_analyzer = live_analyzer
        self._include_frame_image = bool(self.config.get("include_frame_image", False))

    def extract(self, buffer: FrameBuffer) -> SignalBuffer:
        output = self._default_buffer or SignalBuffer()
        self.extract_into(buffer, output)
        return output

    def extract_into(self, buffer: FrameBuffer, output_buffer: SignalBuffer) -> None:
        try:
            for position, frame in enumerate(buffer):
                if output_buffer.closed:
                    buffer.abort()
                    break

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
                    metadata=self._sample_metadata(frame.frame),
                )

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
        if kpts.xy.shape[0] == 0:
            return np.zeros((17, 2)), np.zeros(17)

        xy = kpts.xy[0].cpu().numpy()  # [17,2]
        conf = np.ones(17) if kpts.conf is None else kpts.conf[0].cpu().numpy()  # [17]

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

    def _sample_metadata(self, image: np.ndarray) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": "yolo11-pose",
            "frame_size": (int(image.shape[1]), int(image.shape[0])),
        }
        if self._include_frame_image:
            metadata["frame_image"] = image
        return metadata

    def load_model(self, model_name: str = "yolo11-pose.pt") -> YOLO:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required for YOLO pose extraction. Install it with: pip install ultralytics") from exc

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
