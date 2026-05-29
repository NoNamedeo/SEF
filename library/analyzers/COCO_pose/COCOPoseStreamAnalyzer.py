from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from library.core.artifacts.data.COCOPoseTennisFrameData import COCOPoseTennisFrameData, COCOPoseTennisSequenceData
from library.core.artifacts.signal_sample.COCOSkeletonSignalSample import COCOSkeletonSignalSample
from library.core.artifacts.buffer.DataBuffer import DataBuffer
from library.core.interfaces.BufferContracts import IBuffer
from library.core.interfaces.IData import IData
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalSample import ISignalSample
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.interfaces.StreamingContracts import IStreamingAnalyzer
from library.core.pose.COCOSkeletonNormalizer import (
    COCOSkeletonNormalizationConfig,
    COCOSkeletonNormalizer,
)

DEFAULT_TENNIS_MODEL_PATH = Path(__file__).resolve().parents[2] / "models/skeleton_rf.joblib"


class COCOPoseStreamAnalyzer(IStreamingAnalyzer):
    """Classify tennis movement from normalized COCO skeletons while preserving raw pose data."""

    capabilities = StageCapabilities.streaming(
        stateful=False,
        preserves_order=True,
        realtime_safe=True,
    )

    LABEL_MAP = {
        0: "backhand",
        1: "forehand",
        2: "ready_position",
        3: "serve",
    }

    def __init__(
        self,
        model_path: str | Path = DEFAULT_TENNIS_MODEL_PATH,
        model: Any | None = None,
        normalizer: COCOSkeletonNormalizer | None = None,
        buffer: DataBuffer | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._default_buffer = buffer
        self._retain_frames = bool(self.config.get("retain_frames", False))
        self._include_normalized_skeleton = bool(self.config.get("include_normalized_skeleton", False))
        self._model = model if model is not None else joblib.load(model_path)
        self._normalizer = normalizer or COCOSkeletonNormalizer(
            COCOSkeletonNormalizationConfig.from_mapping(
                self.config.get("skeleton_normalization")
            )
        )

    def analyze(self, signal: ISignal) -> IData:
        output = self._default_buffer or DataBuffer()
        return self.analyze_into(signal, output)

    def analyze_into(
        self,
        signal: Iterable[ISignalSample],
        output_buffer: IBuffer[IData],
    ) -> COCOPoseTennisSequenceData:
        frames: list[COCOPoseTennisFrameData] = []
        frame_count = 0
        try:
            signal_iterator = iter(signal)
            while not output_buffer.closed:
                sample = next(signal_iterator)

                pose_frame = self._map_sample(sample)

                frame_count += 1
                if self._retain_frames:
                    frames.append(pose_frame)
                output_buffer.put(pose_frame)
        except StopIteration:
            pass
        finally:
            if output_buffer.closed:
                self._abort_upstream(signal)
            output_buffer.close()

        return COCOPoseTennisSequenceData(
            frames=frames,
            metadata={
                "frames": frame_count,
                "retained_frames": len(frames),
            },
        )

    def _map_sample(self, sample: ISignalSample) -> COCOPoseTennisFrameData:
        if not isinstance(sample, COCOSkeletonSignalSample):
            raise TypeError(
                "COCOPoseStreamAnalyzer requires COCOSkeletonSignalSample, "
                f"got {type(sample).__name__}."
            )

        raw_skeleton = np.asarray(sample.skeleton, dtype=np.float32)
        normalized = self._normalizer.normalize(raw_skeleton)
        model_features = normalized.skeleton.flatten().reshape(1, -1)
        pred_id = int(self._model.predict(model_features)[0])
        movement = self.LABEL_MAP.get(pred_id, "unknown")
        prediction_confidence = self._prediction_confidence(model_features)

        metadata = dict(sample.metadata)
        frame_image = metadata.pop("frame_image", None)
        metadata.update(
            {
                "predicted_class_id": pred_id,
                "prediction_input": "normalized_coco_skeleton",
                "skeleton_normalization": normalized.metadata(),
            }
        )
        if prediction_confidence is not None:
            metadata["prediction_confidence"] = prediction_confidence
        if self._include_normalized_skeleton:
            metadata["normalized_skeleton"] = normalized.skeleton.copy()

        return COCOPoseTennisFrameData(
            frame_index=sample.frame_index,
            skeleton=raw_skeleton,
            confidence=sample.confidence,
            centroid=sample.centroid,
            tennis_movement=movement,
            timestamp_seconds=sample.timestamp_seconds,
            frame_size=metadata.get("frame_size"),
            frame_image=frame_image,
            metadata=metadata,
        )

    def _prediction_confidence(self, model_features: np.ndarray) -> float | None:
        predict_proba = getattr(self._model, "predict_proba", None)
        if not callable(predict_proba):
            return None
        probabilities = np.asarray(predict_proba(model_features), dtype=np.float32)
        if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
            return None
        return float(np.max(probabilities[0]))

    @staticmethod
    def _abort_upstream(signal: Iterable[ISignalSample]) -> None:
        abort = getattr(signal, "abort", None)
        if callable(abort):
            abort()
