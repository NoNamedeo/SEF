import threading
from pathlib import Path
from types import MethodType

import cv2

from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.artifacts.CompositeFrameCleaner import CompositeFrameCleaner
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.pipeline.Pipeline import Pipeline
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_cleaners.OpenCVMovingAverageCleaner import OpenCVMovingAverageCleaner
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer


def _patch_frame_extractor(extractor: OpenCVBufferedFrameExtractor) -> OpenCVBufferedFrameExtractor:
    def extract(self, frame_cleaners):
        cleaners = frame_cleaners or []
        if isinstance(cleaners, CompositeFrameCleaner):
            cleaners = cleaners.cleaners
        elif not isinstance(cleaners, list):
            cleaners = [cleaners]

        resize = self.config.get("resize")
        gray = bool(self.config.get("gray", False))
        stride = max(1, int(self.config.get("stride", 1)))
        max_frames = self.config.get("max_frames")

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.path}")

        emitted_frames = 0
        read_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if read_index % stride != 0:
                    read_index += 1
                    continue

                if resize:
                    frame = cv2.resize(frame, resize)
                if gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                current_frame = Frame(frame)
                for cleaner in cleaners:
                    cleaned_frame = cleaner.clean(current_frame)
                    current_frame = cleaned_frame if isinstance(cleaned_frame, Frame) else Frame(cleaned_frame)

                self.buffer.put(current_frame)
                emitted_frames += 1
                read_index += 1

                if max_frames is not None and emitted_frames >= max_frames:
                    break
        finally:
            self.buffer.close()
            cap.release()

        return self.buffer

    extractor.extract = MethodType(extract, extractor)
    return extractor


def _patch_signal_extractor(extractor: OpenCVBufferedSignalExtractor) -> OpenCVBufferedSignalExtractor:
    def extract(self, buffer: FrameBuffer):
        results = []
        first_frame = True

        try:
            for frame_number, frame in enumerate(buffer):
                frame_data = frame.frame if isinstance(frame, Frame) else frame

                if first_frame:
                    self.tracker.init(frame_data, tuple(map(int, self.box)))
                    current_box = tuple(map(int, self.box))
                    first_frame = False
                else:
                    success, tracked_box = self.tracker.update(frame_data)
                    if success:
                        x, y, w, h = tracked_box
                        current_box = (int(x), int(y), int(w), int(h))
                        self.box = current_box
                    else:
                        current_box = None

                if current_box:
                    x, y, w, h = current_box
                    centroid = (x + w // 2, y + h // 2)
                else:
                    centroid = None

                results.append(
                    {
                        "frame_number": frame_number,
                        "box": current_box,
                        "centroid": centroid,
                    }
                )

                display_frame = frame_data.copy()
                if len(display_frame.shape) == 2:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

                if current_box:
                    x, y, w, h = current_box
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(
                    display_frame,
                    f"Frame {frame_number}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                cv2.imshow("Tracking", display_frame)
                if cv2.waitKey(30) & 0xFF == 27:
                    break
        finally:
            cv2.destroyAllWindows()

        return Signal(results)

    extractor.extract = MethodType(extract, extractor)
    return extractor


def _build_pipeline(video_path: Path, init_bbox) -> Pipeline:
    pipeline = Pipeline()
    pipeline.frame_extractor = _patch_frame_extractor(
        OpenCVBufferedFrameExtractor(
            buffer=FrameBuffer(buffer_size=256),
            path=str(video_path),
            config={
                "resize": (640, 480),
                "gray": False,
                "stride": 5,
                "max_frames": None,
            },
        )
    )
    pipeline.composite_frame_cleaner = CompositeFrameCleaner([])
    pipeline.signal_extractor = _patch_signal_extractor(
        OpenCVBufferedSignalExtractor(tracker_type="CSRT", start_box=tuple(map(int, init_bbox)))
    )
    pipeline.signal_cleaners = [OpenCVMovingAverageCleaner(window_size=5)]
    pipeline.analyzers = [VerticalPositionAnalyzer()]
    return pipeline


def _run_pipeline(pipeline: Pipeline):
    pipeline._validate_pipeline()

    frame_cleaners = pipeline.composite_frame_cleaner.cleaners if pipeline.composite_frame_cleaner else []
    extraction_thread = threading.Thread(
        target=pipeline.frame_extractor.extract,
        args=(frame_cleaners,),
        daemon=True,
    )
    extraction_thread.start()

    try:
        raw_signal = pipeline.signal_extractor.extract(pipeline.frame_extractor.buffer)
    finally:
        extraction_thread.join()

    raw_data = [analyzer.analyze(raw_signal) for analyzer in pipeline.analyzers]

    cleaned_signal = raw_signal
    for cleaner in pipeline.signal_cleaners:
        cleaned_signal = cleaner.clean(cleaned_signal)

    cleaned_data = [analyzer.analyze(cleaned_signal) for analyzer in pipeline.analyzers]
    return raw_data, cleaned_data


def _read_first_frame(video_path: Path, resize):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Cannot read first frame from video: {video_path}")
    finally:
        cap.release()

    return cv2.resize(frame, resize) if resize else frame


def main():
    root_dir = Path(__file__).resolve().parents[1]
    video_path = root_dir / "videos" / "Crowd.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video NON trovato: {video_path}")

    resize = (640, 480)
    use_interactive_bbox = True

    first_frame = _read_first_frame(video_path, resize)

    if use_interactive_bbox:
        init_bbox = cv2.selectROI("Seleziona oggetto da tracciare", first_frame, False, False)
        cv2.destroyWindow("Seleziona oggetto da tracciare")
    else:
        init_bbox = (300, 200, 80, 120)

    pipeline = _build_pipeline(video_path, init_bbox)
    raw_results, cleaned_results = _run_pipeline(pipeline)

    visualizer = MatplotlibFunctionVisualizer()

    print("Visualizzazione dati grezzi")
    visualizer.visualize(raw_results[0])

    print("Visualizzazione dati puliti")
    visualizer.visualize(cleaned_results[0])


if __name__ == "__main__":
    main()
