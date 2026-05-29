from __future__ import annotations

from pathlib import Path

import numpy as np

from library.core.artifacts.Frame import Frame
from library.core.artifacts.buffer.FrameBuffer import FrameBuffer
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.interfaces.StreamingContracts import IStreamingFrameBufferProcessor
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry, create_builtin_registry
from library.frame_processors.motion_magnification.PhaseMagnificationFrameProcessor import PhaseMagnificationFrameProcessor


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


def _buffer_from_values(values: list[int], *, fps: float = 20.0) -> FrameBuffer:
    buffer = FrameBuffer(buffer_size=len(values) + 1)
    for index, value in enumerate(values):
        image = np.full((8, 8, 3), value, dtype=np.uint8)
        buffer.put(
            Frame(
                image=image,
                index=index,
                timestamp_seconds=index / fps,
                metadata={"source_fps": fps, "tag": f"frame-{index}"},
            )
        )
    buffer.close()
    return buffer


def test_phase_magnification_process_preserves_frame_timeline_and_metadata(monkeypatch) -> None:
    processor = PhaseMagnificationFrameProcessor()

    def fake_magnify(frames, *, fps: float, sampling_rate: float):
        assert fps == 20.0
        assert sampling_rate == 20.0
        return [np.full_like(frame.image, int(frame.image[0, 0, 0]) + 10) for frame in frames]

    monkeypatch.setattr(processor, "_magnify_frames", fake_magnify)
    output = list(processor.process(_buffer_from_values([10, 20, 30])))

    assert [frame.index for frame in output] == [0, 1, 2]
    assert [frame.timestamp_seconds for frame in output] == [0.0, 0.05, 0.1]
    assert [int(frame.image[0, 0, 0]) for frame in output] == [20, 30, 40]
    assert all(frame.metadata["tag"] == f"frame-{index}" for index, frame in enumerate(output))
    assert output[0].metadata["phase_magnification"]["magnification_factor"] == processor.magnification_factor


def test_phase_magnification_is_registered_as_batch_frame_buffer_processor(monkeypatch) -> None:
    processor = PhaseMagnificationFrameProcessor()

    def fake_magnify(frames, *, fps: float, sampling_rate: float):
        return [np.full_like(frame.image, int(frame.image[0, 0, 0]) + 1) for frame in frames]

    monkeypatch.setattr(processor, "_magnify_frames", fake_magnify)
    output = list(processor.process(_buffer_from_values([1, 2, 3])))

    assert not isinstance(processor, IStreamingFrameBufferProcessor)
    assert processor.capabilities.supports_streaming is False
    assert [int(frame.image[0, 0, 0]) for frame in output] == [2, 3, 4]


def test_phase_magnification_generates_matlab_script_with_expected_parameters() -> None:
    tmp_path = Path.cwd() / "output" / "phase_mag_script_test"
    release_dir = Path.cwd() / "external" / "phase_mag" / "Release"
    processor = PhaseMagnificationFrameProcessor(
        magnification_factor=35.0,
        low_cutoff_hz=0.8,
        high_cutoff_hz=2.4,
        sigma=3.0,
        pyr_type="quarterOctave",
        attenuate_other_frequencies=True,
        scale_video=0.5,
        release_dir=release_dir,
    )

    script = processor._matlab_script(
        input_video=tmp_path / "input.avi",
        output_video=tmp_path / "magnified.avi",
        sampling_rate=25.0,
        frame_count=12,
    )

    assert "setPath;" in script
    assert "phaseAmplify(" in script
    assert "35" in script
    assert "0.8" in script
    assert "2.4" in script
    assert "'quarterOctave'" in script
    assert "'attenuateOtherFreq', true" in script
    assert "'scaleVideo', 0.5" in script
    assert "'useFrames', [1 12]" in script


def test_builtin_registry_exposes_motion_magnification_as_frame_buffer_processor() -> None:
    registry = create_builtin_registry()

    processor = registry.create(
        PluginCategory.FRAME_BUFFER_PROCESSOR,
        "motion_magnification",
        magnification_factor=15.0,
    )

    assert isinstance(processor, PhaseMagnificationFrameProcessor)


def test_config_builder_creates_motion_magnification_frame_buffer_processor() -> None:
    registry = PluginRegistry()
    registry.register(PluginCategory.FRAME_EXTRACTOR, "static", StaticFrameExtractor)
    registry.register(PluginCategory.SIGNAL_EXTRACTOR, "passthrough", PassthroughSignalExtractor)
    registry.register(PluginCategory.ANALYZER, "constant", ConstantAnalyzer)
    registry.register(PluginCategory.FRAME_BUFFER_PROCESSOR, "motion_magnification", PhaseMagnificationFrameProcessor)

    context = ConfigPipelineBuilder(registry).build_context(
        {
            "pipeline": {
                "frame_extractor": {"name": "static"},
                "frame_processors": [
                    {
                        "name": "motion_magnification",
                        "processor_type": "frame_buffer",
                        "params": {"magnification_factor": 12.0},
                    }
                ],
                "signal_extractor": {"name": "passthrough"},
                "analyzers": [{"name": "constant"}],
            }
        }
    )

    assert isinstance(context.frame_processors[0], PhaseMagnificationFrameProcessor)
