from __future__ import annotations

import unittest

import cv2
import numpy as np

from sef.builtin.frame_processors.ColorStabilizationFrameProcessor import ColorStabilizationFrameProcessor
from sef.core.artifacts.Frame import Frame


class ColorStabilizationFrameProcessorTests(unittest.TestCase):
    def test_processor_reduces_temporal_brightness_variance(self) -> None:
        frames = [Frame(image=image, index=index, timestamp_seconds=index / 30.0) for index, image in enumerate(self._flickering_sequence())]
        processor = ColorStabilizationFrameProcessor(
            color_space="LAB",
            techniques=(
                "luminance_normalization",
                "histogram_normalization",
                "gamma_correction",
                "temporal_smoothing",
            ),
            stabilization_strength=1.0,
            temporal_alpha=1.0,
            luminance_max_shift=120.0,
            histogram_max_gain=2.0,
            gamma_limits=(0.50, 2.20),
            chroma_strength=0.35,
        )

        original_variance = float(np.var([self._mean_luminance(frame.frame) for frame in frames]))
        processed_frames = [processor.process(frame) for frame in frames]
        processed_variance = float(np.var([self._mean_luminance(frame.frame) for frame in processed_frames]))

        self.assertLess(processed_variance, original_variance * 0.10)
        self.assertTrue(all(frame.frame.dtype == np.uint8 for frame in processed_frames))
        self.assertTrue(all(frame.frame.shape == frames[0].frame.shape for frame in processed_frames))

    def test_processor_supports_configured_color_spaces(self) -> None:
        image = self._base_image()

        for color_space in ("RGB", "HSV", "LAB", "YCrCb"):
            with self.subTest(color_space=color_space):
                processor = ColorStabilizationFrameProcessor(
                    color_space=color_space,
                    techniques=("clahe", "luminance_normalization", "gamma_correction", "temporal_smoothing"),
                    stabilization_strength=0.65,
                )

                processed = processor.process(Frame(image=image, index=3, timestamp_seconds=0.1))

                self.assertEqual(processed.frame.shape, image.shape)
                self.assertEqual(processed.frame.dtype, np.uint8)
                self.assertIn("color_stabilization", processed.metadata)

    def test_processor_emits_optional_comparison_and_intermediate_artifacts(self) -> None:
        image = self._base_image(width=32, height=24)
        processor = ColorStabilizationFrameProcessor(
            color_space="LAB",
            techniques=("luminance_normalization", "temporal_smoothing"),
            emit_comparison_overlay=True,
            emit_intermediate_artifacts=True,
        )

        processed = processor.process(Frame(image=image, index=1, timestamp_seconds=0.03, metadata={"source": "unit-test"}))
        artifact_payload = processed.metadata["color_stabilization"]

        self.assertEqual(processed.metadata["source"], "unit-test")
        self.assertIn("metrics", artifact_payload)
        self.assertIn("comparison_overlay", artifact_payload)
        self.assertIn("intermediate_artifacts", artifact_payload)
        self.assertEqual(artifact_payload["comparison_overlay"].shape[:2], (24, 64))
        self.assertEqual(artifact_payload["intermediate_artifacts"]["luminance_before"].shape, (24, 32))
        self.assertEqual(artifact_payload["intermediate_artifacts"]["luminance_after"].shape, (24, 32))

    @staticmethod
    def _mean_luminance(image: np.ndarray) -> float:
        if image.ndim == 2:
            return float(np.mean(image))
        return float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]))

    @staticmethod
    def _flickering_sequence() -> list[np.ndarray]:
        base = ColorStabilizationFrameProcessorTests._base_image()
        factors = (0.58, 1.38, 0.72, 1.26, 0.64, 1.45, 0.80, 1.18)
        color_offsets = (
            (8.0, -4.0, 3.0),
            (-6.0, 5.0, -2.0),
            (3.0, -6.0, 4.0),
            (-4.0, 4.0, -5.0),
            (6.0, -3.0, 2.0),
            (-7.0, 5.0, -3.0),
            (2.0, -5.0, 3.0),
            (-3.0, 3.0, -4.0),
        )
        return [
            np.clip((base.astype(np.float32) * factor) + np.array(offset, dtype=np.float32), 0, 255).astype(np.uint8)
            for factor, offset in zip(factors, color_offsets, strict=True)
        ]

    @staticmethod
    def _base_image(width: int = 96, height: int = 64) -> np.ndarray:
        x_gradient = np.linspace(35, 205, width, dtype=np.float32)
        y_gradient = np.linspace(15, 55, height, dtype=np.float32)[:, None]
        luminance = np.clip(x_gradient + y_gradient, 0, 255)
        texture = ((np.indices((height, width)).sum(axis=0) % 8) * 4).astype(np.float32)
        blue = np.clip(luminance * 0.72 + texture, 0, 255)
        green = np.clip(luminance * 0.92 + (texture * 0.5), 0, 255)
        red = np.clip(luminance * 1.06 - texture, 0, 255)
        image = np.dstack([blue, green, red]).astype(np.uint8)
        cv2.rectangle(image, (width // 5, height // 4), (width // 2, height // 2), (220, 180, 120), 2)
        cv2.circle(image, (width * 3 // 4, height * 2 // 3), max(4, width // 10), (60, 180, 240), -1)
        return image


if __name__ == "__main__":
    unittest.main()
