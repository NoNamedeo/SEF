# Getting Started

This guide builds and runs a complete pipeline through the public `sef` facade.
The example is intentionally small so plugin authors can focus on component
contracts without touching the lower-level runtime.

## Install Context

Install SEF in editable mode, then run examples from the repository root:

```bash
pip install -e .
python -m examples.minimal_pipeline
```

The base install supports the public API, core runtime, config parsing, and
minimal custom pipelines. Install adapter extras only when you use those
features:

```bash
pip install -e ".[opencv]"        # OpenCV video/tracking/ArUco components
pip install -e ".[visualization]" # Matplotlib visualizers
pip install -e ".[ui]"            # Streamlit Studio
pip install -e ".[yolo]"          # Ultralytics pose extraction
pip install -e ".[pose]"          # COCO pose analyzer model helpers
```

Validate a YAML config from the terminal:

```bash
sef init tracking-demo
sef doctor --config pipeline.yaml
sef validate pipeline.yaml
sef run pipeline.yaml --dry-run --explain
sef run pipeline.yaml --output outputs/run-001
sef components list
sef components inspect vertical_position
```

`sef init tracking-demo` creates a video-based demo config that expects the user
to place a video at `videos/input.mp4` and install `sef[opencv]`; it does not
download or include assets.
`sef doctor` exits with `0` when only warnings are found and exits with `1` only
for blocking errors.

## Minimal Pipeline

```python
import sef

outputs = (
    sef.pipeline("docs-quickstart")
    .frames(DemoFrameExtractor, frame_count=3)
    .signals(DemoSignalExtractor)
    .analyze(SampleCountAnalyzer)
    .visualize(SummaryVisualizer)
    .run()
)

print(outputs.results[0].y)
print(outputs.final_artifacts[0].content)
```

The complete runnable file is available in the repository at
`examples/minimal_pipeline.py`.

## Expected Output

The analyzer emits one `TwoDimGraphData` result with the sample count. The
visualizer emits one `TextArtifact`.

```text
results: 1
artifacts: 1
sample_count: 3.0
summary: Sample count: 3.0
```

## Error Handling

Catch configuration errors separately from execution errors:

```python
import sef
from sef.core import PipelineConfigurationError, PipelineExecutionError

try:
    config = sef.load_config("pipeline.yaml")
    outputs = sef.from_config(config).run()
except PipelineConfigurationError as exc:
    print(exc.path, exc)
except PipelineExecutionError as exc:
    print(exc.stage, exc.cause)
```

Configuration errors usually indicate invalid config shape, unknown plugins, or
invalid plugin constructor parameters. Execution errors indicate a stage failed
after a valid context was built.
