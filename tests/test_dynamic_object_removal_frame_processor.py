from __future__ import annotations

import numpy as np
import pytest

from sef.core.artifacts.Frame import Frame
from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.interfaces.IAnalyzer import IAnalyzer
from sef.core.interfaces.IFrameExtractor import IFrameExtractor
from sef.core.interfaces.ISignalExtractor import ISignalExtractor
from sef.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from sef.core.pipeline.FrameProcessingStage import FrameProcessorExecutionContext
from sef.core.pipeline.IntermediateFrameCapture import (
    IntermediateFrameArtifactStore,
    IntermediateFrameCaptureConfig,
)
from sef.core.plugins.PluginRegistry import PluginCategory, PluginRegistry, create_builtin_registry
from sef.builtin.frame_processors.dynamic_object_removal import DynamicObjectRemovalFrameProcessor


class StaticFrameExtractor(IFrameExtractor):
    def extract(self) -> FrameBuffer:
        buffer = FrameBuffer(buffer_size=1)
        buffer.close()
        return buffer


class PassthroughSignalExtractor(ISignalExtractor):
    def extract(self, buffer):
        return []


class ConstantAnalyzer(IAnalyzer):
    def analyze(self, signal):
        return []


def _buffer_from_images(images: list[np.ndarray]) -> FrameBuffer:
    buffer = FrameBuffer(buffer_size=len(images) + 1)
    for index, image in enumerate(images):
        buffer.put(Frame(image=image, index=index, timestamp_seconds=index / 10.0))
    buffer.close()
    return buffer


def _read_images(buffer: FrameBuffer) -> list[np.ndarray]:
    return [frame.image for frame in buffer]


def _background() -> np.ndarray:
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[:, :, 0] = 30
    image[:, :, 1] = 70
    image[:, :, 2] = 120
    return image


def _processor(**overrides) -> DynamicObjectRemovalFrameProcessor:
    params = {
        "sampling_stride": 1,
        "max_sampled_frames": 8,
        "difference_threshold": 20,
        "morph_kernel_size": 1,
        "opening_iterations": 0,
        "closing_iterations": 0,
        "dilation_iterations": 0,
        "min_component_area": 1,
        "max_processed_frames": 20,
    }
    params.update(overrides)
    return DynamicObjectRemovalFrameProcessor(**params)


def test_dynamic_object_removal_replaces_only_transient_pixels_with_temporal_background() -> None:
    background = _background()
    frames = []
    positions = [(0, 0), (0, 5), (5, 0), (5, 5), (9, 9)]
    for y, x in positions:
        image = background.copy()
        image[y : y + 2, x : x + 2] = (255, 255, 255)
        frames.append(image)

    processed = _read_images(_processor().process(_buffer_from_images(frames)))

    assert len(processed) == len(frames)
    assert all(np.array_equal(image, background) for image in processed)


def test_dynamic_object_removal_preserves_protected_mask_pixels() -> None:
    background = _background()
    frames = []
    for index in range(5):
        image = background.copy()
        if index == 2:
            image[:3, :3] = (255, 255, 255)
            image[6:9, 6:9] = (255, 255, 255)
        frames.append(image)

    protected_mask = np.zeros(background.shape[:2], dtype=np.bool_)
    protected_mask[:3, :3] = True
    output = list(_processor(protected_mask=protected_mask).process(_buffer_from_images(frames)))

    assert np.array_equal(output[2].image[:3, :3], frames[2][:3, :3])
    assert np.array_equal(output[2].image[6:9, 6:9], background[6:9, 6:9])
    assert output[2].metadata["dynamic_object_removal"]["removed_pixel_count"] == 9


def test_dynamic_object_removal_supports_grayscale_frames() -> None:
    background = np.full((8, 8), 40, dtype=np.uint8)
    frames = []
    for index in range(5):
        image = background.copy()
        image[index : index + 2, index : index + 2] = 240
        frames.append(image)

    processed = _read_images(_processor().process(_buffer_from_images(frames)))

    assert all(image.ndim == 2 for image in processed)
    assert all(np.array_equal(image, background) for image in processed)


def test_dynamic_object_removal_rejects_inconsistent_frame_shapes() -> None:
    frames = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((5, 4, 3), dtype=np.uint8),
    ]

    with pytest.raises(ValueError, match="shape"):
        _processor().process(_buffer_from_images(frames))


def test_dynamic_object_removal_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="sampling_stride"):
        DynamicObjectRemovalFrameProcessor(sampling_stride=0)

    with pytest.raises(ValueError, match="morph_kernel_size"):
        DynamicObjectRemovalFrameProcessor(morph_kernel_size=4)


def test_dynamic_object_removal_emits_bounded_intermediate_artifacts() -> None:
    background = _background()
    frames = []
    for index in range(3):
        image = background.copy()
        image[2:5, 2 + index:5 + index] = (255, 255, 255)
        frames.append(image)

    store = IntermediateFrameArtifactStore(
        IntermediateFrameCaptureConfig(
            enabled=True,
            sampling_interval=1,
            max_stored_frames=2,
            include_original=True,
        )
    )
    context = FrameProcessorExecutionContext(
        processor_index=0,
        processor_name="DynamicObjectRemovalFrameProcessor",
        stage_name="frame_processing[0].DynamicObjectRemovalFrameProcessor",
        intermediate_store=store,
    )

    _processor(emit_intermediate_artifacts=True).process_with_context(_buffer_from_images(frames), context)
    collection = store.to_collection()

    assert collection.count == 2
    artifact = collection.artifacts[-1]
    assert artifact.original_frame is not None
    assert {mask.label for mask in artifact.masks} >= {
        "raw_dynamic_mask",
        "refined_dynamic_mask",
        "effective_removal_mask",
    }
    assert artifact.overlays[0].label == "estimated_background"


def test_builtin_registry_exposes_dynamic_object_removal_as_frame_buffer_processor() -> None:
    registry = create_builtin_registry()

    processor = registry.create(
        PluginCategory.FRAME_BUFFER_PROCESSOR,
        "dynamic_object_removal",
        max_processed_frames=5,
    )

    assert isinstance(processor, DynamicObjectRemovalFrameProcessor)


def test_config_builder_creates_dynamic_object_removal_frame_buffer_processor() -> None:
    registry = PluginRegistry()
    registry.register(PluginCategory.FRAME_EXTRACTOR, "static", StaticFrameExtractor)
    registry.register(PluginCategory.SIGNAL_EXTRACTOR, "passthrough", PassthroughSignalExtractor)
    registry.register(PluginCategory.ANALYZER, "constant", ConstantAnalyzer)
    registry.register(PluginCategory.FRAME_BUFFER_PROCESSOR, "dynamic_object_removal", DynamicObjectRemovalFrameProcessor)

    context = ConfigPipelineBuilder(registry).build_context(
        {
            "pipeline": {
                "frame_extractor": {"name": "static"},
                "frame_processors": [
                    {
                        "name": "dynamic_object_removal",
                        "processor_type": "frame_buffer",
                        "params": {"max_processed_frames": 5},
                    }
                ],
                "signal_extractor": {"name": "passthrough"},
                "analyzers": [{"name": "constant"}],
            }
        }
    )

    assert isinstance(context.frame_processors[0], DynamicObjectRemovalFrameProcessor)
