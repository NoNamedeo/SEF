from __future__ import annotations

from pathlib import Path

from sef.cli.constants import (
    DEFAULT_CONFIG_NAME,
    OUTPUT_DIR_NAME,
    PLUGIN_DIR_NAME,
    SCAFFOLD_FILE_MARKER,
    VIDEO_DIR_NAME,
)


class ProjectScaffolder:
    """Creates the files and directories for a starter SEF project."""

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root or Path.cwd())

    def create(self, *, template: str = "default", force: bool = False) -> tuple[list[Path], list[Path]]:
        """Create scaffold files and return created and skipped absolute paths."""
        created: list[Path] = []
        skipped: list[Path] = []

        for directory in (self._root / PLUGIN_DIR_NAME, self._root / VIDEO_DIR_NAME, self._root / OUTPUT_DIR_NAME):
            if directory.exists():
                skipped.append(directory)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)

        for relative_path, content in scaffold_files(template).items():
            target = self._root / relative_path
            if target.exists() and not can_overwrite_scaffold(target, force=force):
                skipped.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            created.append(target)

        return created, skipped


def can_overwrite_scaffold(path: Path, *, force: bool) -> bool:
    """Return True when a file can be overwritten by `sef init --force`."""
    if not path.exists():
        return True
    if not force:
        return False
    try:
        return SCAFFOLD_FILE_MARKER in path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False


def scaffold_files(template: str) -> dict[str, str]:
    """Return scaffold file contents for the requested template."""
    if template == "plugin":
        return {
            DEFAULT_CONFIG_NAME: _plugin_pipeline_yaml(),
            "README.md": _plugin_readme(),
            "plugins/__init__.py": f"{SCAFFOLD_FILE_MARKER}\n",
            "plugins/custom_components.py": _plugin_components_file(),
            "tests/test_custom_components.py": _plugin_test_file(),
        }
    if template == "tracking-demo":
        return {
            DEFAULT_CONFIG_NAME: _tracking_demo_pipeline_yaml(),
            "README.md": _tracking_demo_readme(),
            "plugins/example_local_plugins.py": _example_plugin_file(),
        }
    return {
        DEFAULT_CONFIG_NAME: _default_pipeline_yaml(),
        "README.md": _default_readme(),
        "plugins/example_local_plugins.py": _example_plugin_file(),
    }


def _default_pipeline_yaml() -> str:
    return f"""{SCAFFOLD_FILE_MARKER}
schema_version: "1.0"
run:
  runtime:
    frame_buffer_size: 8
    signal_buffer_size: 8
    data_buffer_size: 8

pipeline:
  frame_extractor:
    name: opencv_buffered
    params:
      path: videos/input.mp4
      config:
        max_frames: 300

  frame_processors: []

  signal_extractor:
    name: opencv_tracker
    params:
      start_box: [100, 100, 80, 80]
      tracker_type: MIL

  signal_cleaners:
    - name: moving_average
      params:
        window_size: 5

  analyzers:
    - name: vertical_position

  visualizers:
    - name: matplotlib
      result_indices: [0]
"""


def _tracking_demo_pipeline_yaml() -> str:
    return f"""{SCAFFOLD_FILE_MARKER}
# Put your demo video at videos/input.mp4 before running this pipeline.
schema_version: "1.0"
run:
  runtime:
    frame_buffer_size: 8
    signal_buffer_size: 8
    data_buffer_size: 8

pipeline:
  frame_extractor:
    name: opencv_buffered
    params:
      path: videos/input.mp4
      config:
        max_frames: 300
        stride: 1

  frame_processors:
    - name: opencv_gray
      processor_type: single_frame

  signal_extractor:
    name: opencv_tracker
    params:
      start_box: [100, 100, 80, 80]
      tracker_type: MIL

  signal_cleaners:
    - name: moving_average
      params:
        window_size: 5

  analyzers:
    - name: vertical_position

  visualizers:
    - name: matplotlib
      result_indices: [0]

  intermediate_frames:
    enabled: false
"""


def _plugin_pipeline_yaml() -> str:
    return f"""{SCAFFOLD_FILE_MARKER}
schema_version: "1.0"
pipeline:
  frame_extractor:
    name: demo_frames
    params:
      frame_count: 5

  frame_processors:
    - name: tag_frames
      processor_type: frame_buffer
      params:
        label: scaffolded

  signal_extractor:
    name: demo_signal

  analyzers:
    - name: sample_count
      params:
        scale: 1.0

  visualizers:
    - name: summary_text
      result_indices: [0]
"""


def _default_readme() -> str:
    return f"""{SCAFFOLD_FILE_MARKER}
# SEF project

## Usage

1. Put an input video at `videos/input.mp4`.
2. Validate the pipeline:

```bash
sef validate pipeline.yaml --strict
```

3. Inspect the execution plan without running:

```bash
sef run pipeline.yaml --dry-run --explain
```

4. Run and collect outputs:

```bash
sef run pipeline.yaml --output outputs/run-001
```

Local plugins can be added as `.py` files under `plugins/` and registered with
decorators such as `@sef.analyzer("my_analyzer")`.
"""


def _tracking_demo_readme() -> str:
    return f"""{SCAFFOLD_FILE_MARKER}
# SEF tracking demo

This scaffold is video-based but intentionally does not include media assets.
Place your video at:

```text
videos/input.mp4
```

Then run:

```bash
sef validate pipeline.yaml --strict
sef run pipeline.yaml --dry-run --explain
sef run pipeline.yaml --output outputs/tracking-demo
```

The default `start_box` in `pipeline.yaml` is `[100, 100, 80, 80]`. Adjust it to
match the object you want to track in your video.
"""


def _plugin_readme() -> str:
    return f"""{SCAFFOLD_FILE_MARKER}
# SEF plugin project

This scaffold is intentionally small and dependency-light. It shows how to
register local function plugins with metadata, aliases, and capabilities.

## Run

```bash
sef components inspect sample_count
sef validate pipeline.yaml --strict
sef run pipeline.yaml --dry-run --explain
sef run pipeline.yaml --output outputs/plugin-demo
```

## Test

```bash
python -m pytest tests/test_custom_components.py
```

Local plugin modules under `plugins/*.py` are imported by the SEF CLI before
validation, execution, and component inspection.
"""


def _example_plugin_file() -> str:
    return f'''{SCAFFOLD_FILE_MARKER}
from __future__ import annotations

import sef


@sef.analyzer("example_sample_counter")
def sample_counter(signal):
    """Return a simple count-like result for local plugin experiments."""
    return signal
'''


def _plugin_components_file() -> str:
    return f'''{SCAFFOLD_FILE_MARKER}
from __future__ import annotations

import numpy as np

import sef
from sef.core.artifacts import Frame, Signal
from sef.core.artifacts.buffer import FrameBuffer
from sef.core.artifacts.data import TwoDimGraphData
from sef.core.artifacts.signal_sample import BoxSignalSample
from sef.core.interfaces import StageCapabilities
from sef.core.visualization import TextArtifact


@sef.frame_extractor(
    "demo_frames",
    description="Create deterministic demo frames for plugin development.",
    metadata={{"domain": "demo", "output": "FrameBuffer"}},
)
def demo_frames(frame_count: int = 5) -> FrameBuffer:
    buffer = FrameBuffer(frame_count)
    for index in range(frame_count):
        buffer.put(
            Frame(
                image=np.zeros((4, 4, 3), dtype=np.uint8),
                index=index,
                timestamp_seconds=float(index),
                metadata={{"source": "plugin-scaffold"}},
            )
        )
    buffer.close()
    return buffer


@sef.frame_buffer_processor(
    "tag_frames",
    description="Annotate each frame with a metadata label.",
    metadata={{"domain": "demo", "input": "FrameBuffer", "output": "FrameBuffer"}},
    capabilities=StageCapabilities.batch(stateful=False),
)
def tag_frames(buffer: FrameBuffer, label: str = "processed") -> FrameBuffer:
    output = buffer.clone_empty()
    for frame in buffer:
        output.put(
            Frame(
                image=frame.image,
                index=frame.index,
                timestamp_seconds=frame.timestamp_seconds,
                metadata={{**frame.metadata, "label": label}},
            )
        )
    output.close()
    return output


@sef.signal_extractor(
    "demo_signal",
    description="Convert frames into one centroid sample per frame.",
    metadata={{"domain": "demo", "input": "FrameBuffer", "output": "Signal"}},
)
def demo_signal(buffer: FrameBuffer) -> Signal:
    return Signal(
        [
            BoxSignalSample(
                frame_index=int(frame.index or 0),
                box=(0, 0, 2, 2),
                centroid=(1.0, float(frame.index or 0)),
                timestamp_seconds=frame.timestamp_seconds,
                metadata=dict(frame.metadata),
            )
            for frame in buffer
        ]
    )


@sef.analyzer(
    "sample_count",
    description="Count samples in a signal.",
    aliases=("count_samples",),
    metadata={{
        "domain": "demo",
        "input": "Signal",
        "output": "TwoDimGraphData",
        "params": {{"scale": {{"type": "float", "default": 1.0}}}},
    }},
)
def sample_count(signal: Signal, scale: float = 1.0) -> TwoDimGraphData:
    count = float(len(list(signal))) * scale
    return TwoDimGraphData(x=[0.0], y=[count], title="Sample count")


@sef.visualizer(
    "summary_text",
    description="Render analyzer output as a text artifact.",
    metadata={{"domain": "demo", "input": "TwoDimGraphData", "output": "TextArtifact"}},
)
def summary_text(data: TwoDimGraphData) -> TextArtifact:
    return TextArtifact(
        kind="text",
        title="Summary",
        content=f"Sample count: {{data.y[0]}}",
    )
'''


def _plugin_test_file() -> str:
    return f'''{SCAFFOLD_FILE_MARKER}
from __future__ import annotations

import plugins.custom_components  # noqa: F401 - imports decorator registrations.
import sef


def test_scaffolded_pipeline_runs() -> None:
    outputs = (
        sef.pipeline("plugin-test", include_builtins=False)
        .frames("demo_frames", frame_count=3)
        .process("tag_frames", processor_type="frame_buffer", label="tested")
        .signals("demo_signal")
        .analyze("sample_count", scale=2.0)
        .visualize("summary_text")
        .run()
    )

    assert outputs.results[0].y == [6.0]
    assert outputs.final_artifacts[0].content == "Sample count: 6.0"
'''
