Signal extraction framework for Python, for analyzing videos and sequences of images with computer vision techniques.

## Core pipeline

The library now exposes a coherent synchronous pipeline:

`FrameExtractor -> SignalExtractor -> SignalCleaner* -> Analyzer+`

Minimal usage:

```python
from library.core.pipeline.PipelineBuilder import PipelineBuilder
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
from library.signal_cleaners.OpenCVMovingAverageCleaner import OpenCVMovingAverageCleaner
from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer

pipeline = (
    PipelineBuilder()
    .with_frame_extractor(OpenCVBufferedFrameExtractor("videos/Crowd.mp4"))
    .with_signal_extractor(
        OpenCVBufferedSignalExtractor(start_box=(300, 200, 80, 120))
    )
    .add_signal_cleaner(OpenCVMovingAverageCleaner(window_size=5))
    .add_analyzer(VerticalPositionAnalyzer())
    .build()
)

results = pipeline.run()
```

## Plugin registry

The core also exposes a lightweight registry for pluggable components:

```python
from library.core.plugins import create_builtin_registry

registry = create_builtin_registry()
signal_extractor = registry.create(
    "signal_extractor",
    "opencv_tracker",
    start_box=(300, 200, 80, 120),
)
```

## Tests

Run the integration and registry tests with:

```bash
python3 -m unittest discover -s tests -v
```
