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

  runtime:
    frame_buffer_size: 8
    signal_buffer_size: 8
    data_buffer_size: 8
"""


def _tracking_demo_pipeline_yaml() -> str:
    return f"""{SCAFFOLD_FILE_MARKER}
# Put your demo video at videos/input.mp4 before running this pipeline.
schema_version: "1.0"
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

  runtime:
    frame_buffer_size: 8
    signal_buffer_size: 8
    data_buffer_size: 8
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


def _example_plugin_file() -> str:
    return f'''{SCAFFOLD_FILE_MARKER}
from __future__ import annotations

import sef


@sef.analyzer("example_sample_counter")
def sample_counter(signal):
    """Return a simple count-like result for local plugin experiments."""
    return signal
'''
