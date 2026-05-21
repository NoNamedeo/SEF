from __future__ import annotations

import numpy as np

from library.analyzers.COCOPoseStreamAnalyzer import COCOPoseStreamAnalyzer
from library.core.artifacts.COCOPoseFrameData import COCOPoseFrameData, COCOPoseSequenceData
from library.core.artifacts.COCOSkeletonSignalSample import COCOSkeletonSignalSample
from library.core.artifacts.DataBuffer import DataBuffer
from library.core.artifacts.Signal import Signal
from library.visualizers.OpenCVCOCOPoseRealtimeVisualizer import OpenCVCOCOPoseRealtimeVisualizer


def test_coco_pose_stream_analyzer_maps_skeleton_samples_to_pose_frames() -> None:
    skeleton = np.zeros((17, 2), dtype=float)
    skeleton[5] = (10.0, 20.0)
    confidence = np.ones(17, dtype=float)
    frame_image = np.full((480, 640, 3), 32, dtype=np.uint8)
    signal = Signal(
        [
            COCOSkeletonSignalSample(
                frame_index=7,
                skeleton=skeleton,
                confidence=confidence,
                centroid=(30.0, 40.0),
                timestamp_seconds=1.5,
                metadata={"frame_size": (640, 480), "frame_image": frame_image},
            )
        ]
    )
    output = DataBuffer(buffer_size=2, consumers=0)

    result = COCOPoseStreamAnalyzer(config={"retain_frames": True}).analyze_into(signal, output)

    assert isinstance(result, COCOPoseSequenceData)
    assert len(result.frames) == 1
    assert result.metadata == {"frames": 1, "retained_frames": 1}
    frame = result.frames[0]
    assert frame.frame_index == 7
    assert frame.frame_size == (640, 480)
    assert frame.centroid == (30.0, 40.0)
    np.testing.assert_array_equal(frame.skeleton, skeleton)
    np.testing.assert_array_equal(frame.confidence, confidence)
    np.testing.assert_array_equal(frame.frame_image, frame_image)
    assert "frame_image" not in frame.metadata


def test_coco_pose_stream_analyzer_does_not_retain_frames_by_default() -> None:
    skeleton = np.zeros((17, 2), dtype=float)
    confidence = np.ones(17, dtype=float)
    frame_image = np.zeros((32, 32, 3), dtype=np.uint8)
    samples = [
        COCOSkeletonSignalSample(
            frame_index=index,
            skeleton=skeleton,
            confidence=confidence,
            metadata={"frame_size": (32, 32), "frame_image": frame_image},
        )
        for index in range(3)
    ]
    output = DataBuffer(buffer_size=2, consumers=0)

    result = COCOPoseStreamAnalyzer().analyze_into(Signal(samples), output)

    assert result.frames == []
    assert result.metadata == {"frames": 3, "retained_frames": 0}


def test_coco_pose_stream_analyzer_aborts_upstream_when_output_is_closed() -> None:
    output = DataBuffer(buffer_size=2, consumers=0)
    output.abort()
    signal = AbortableSignal([])

    result = COCOPoseStreamAnalyzer().analyze_into(signal, output)

    assert signal.aborted is True
    assert result.metadata == {"frames": 0, "retained_frames": 0}


def test_opencv_coco_pose_realtime_visualizer_consumes_pose_stream(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("cv2.namedWindow", lambda *_args, **_kwargs: calls.append("namedWindow"))
    monkeypatch.setattr("cv2.imshow", lambda *_args, **_kwargs: calls.append("imshow"))
    monkeypatch.setattr("cv2.waitKey", lambda *_args, **_kwargs: -1)
    monkeypatch.setattr("cv2.destroyWindow", lambda *_args, **_kwargs: calls.append("destroyWindow"))

    skeleton = np.zeros((17, 2), dtype=float)
    skeleton[5] = (10.0, 10.0)
    skeleton[6] = (30.0, 10.0)
    confidence = np.ones(17, dtype=float)
    pose_frame = COCOPoseFrameData(
        frame_index=1,
        skeleton=skeleton,
        confidence=confidence,
        centroid=(20.0, 20.0),
        frame_size=(64, 48),
    )

    artifacts = OpenCVCOCOPoseRealtimeVisualizer(config={"window_name": "test"}).render_stream([pose_frame])

    assert artifacts == ()
    assert calls == ["namedWindow", "imshow", "destroyWindow"]


def test_opencv_coco_pose_realtime_visualizer_draws_over_source_frame(monkeypatch) -> None:
    rendered_frames: list[np.ndarray] = []

    monkeypatch.setattr("cv2.namedWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("cv2.imshow", lambda _window_name, image: rendered_frames.append(image.copy()))
    monkeypatch.setattr("cv2.waitKey", lambda *_args, **_kwargs: -1)
    monkeypatch.setattr("cv2.destroyWindow", lambda *_args, **_kwargs: None)

    source_frame = np.full((48, 64, 3), (42, 43, 44), dtype=np.uint8)
    skeleton = np.zeros((17, 2), dtype=float)
    confidence = np.zeros(17, dtype=float)
    pose_frame = COCOPoseFrameData(
        frame_index=1,
        skeleton=skeleton,
        confidence=confidence,
        frame_size=(64, 48),
        frame_image=source_frame,
    )

    OpenCVCOCOPoseRealtimeVisualizer(config={"window_name": "test"}).render_stream([pose_frame])

    assert len(rendered_frames) == 1
    np.testing.assert_array_equal(rendered_frames[0][-1, -1], source_frame[-1, -1])


def test_opencv_coco_pose_realtime_visualizer_aborts_stream_on_escape(monkeypatch) -> None:
    monkeypatch.setattr("cv2.namedWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("cv2.imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("cv2.waitKey", lambda *_args, **_kwargs: 27)
    monkeypatch.setattr("cv2.destroyWindow", lambda *_args, **_kwargs: None)

    pose_frame = COCOPoseFrameData(
        frame_index=1,
        skeleton=np.zeros((17, 2), dtype=float),
        confidence=np.ones(17, dtype=float),
        frame_size=(64, 48),
    )
    stream = AbortableDataStream([pose_frame])

    OpenCVCOCOPoseRealtimeVisualizer(config={"window_name": "test"}).render_stream(stream)

    assert stream.aborted is True


class AbortableSignal:
    def __init__(self, samples):
        self._samples = samples
        self.aborted = False

    def __iter__(self):
        return iter(self._samples)

    def abort(self) -> None:
        self.aborted = True


class AbortableDataStream:
    def __init__(self, items):
        self._items = items
        self.aborted = False

    def __iter__(self):
        return iter(self._items)

    def abort(self) -> None:
        self.aborted = True
