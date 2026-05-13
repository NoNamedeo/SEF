from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.IFrameExporter import FrameExportContext
from library.core.visualization.VisualArtifact import ArtifactRole, VideoFileArtifact
from library.exporters.OpenCVFrameBufferVideoExporter import OpenCVFrameBufferVideoExporter


def test_frame_buffer_video_exporter_writes_file_artifact_and_preserves_stream(tmp_path: Path) -> None:
    output_path = tmp_path / "processed.mp4"
    exporter = OpenCVFrameBufferVideoExporter(output_path, fps=12.0)

    result = exporter.export(
        _frame_buffer(
            [
                _frame(0, fill_value=20),
                _frame(1, fill_value=80),
                _frame(2, fill_value=140),
            ]
        ),
        FrameExportContext(
            pipeline_id="test-pipeline",
            exporter_name="OpenCVFrameBufferVideoExporter",
            execution_metadata={},
        ),
    )

    output_frames = list(result.buffer)

    assert [frame.index for frame in output_frames] == [0, 1, 2]
    assert len(result.artifacts) == 1
    assert isinstance(result.artifacts[0], VideoFileArtifact)
    assert result.artifacts[0].role is ArtifactRole.FINAL_OUTPUT
    assert result.artifacts[0].path == output_path
    assert _encoded_frame_count(output_path) == 3


def test_frame_buffer_video_exporter_rejects_empty_buffers(tmp_path: Path) -> None:
    exporter = OpenCVFrameBufferVideoExporter(tmp_path / "empty.mp4", fps=12.0)

    with pytest.raises(ValueError, match="empty frame buffer"):
        exporter.export(
            _frame_buffer([]),
            FrameExportContext(
                pipeline_id="test-pipeline",
                exporter_name="OpenCVFrameBufferVideoExporter",
                execution_metadata={},
            ),
        )


def _frame(index: int, *, fill_value: int) -> Frame:
    image = np.full((12, 16, 3), fill_value, dtype=np.uint8)
    return Frame(image=image, index=index, timestamp_seconds=index / 12.0)


def _frame_buffer(frames: list[Frame]) -> FrameBuffer:
    buffer = FrameBuffer(buffer_size=len(frames))
    for frame in frames:
        buffer.put(frame)
    buffer.close()
    return buffer


def _encoded_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
