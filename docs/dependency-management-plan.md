# Dependency Management Plan

SEF keeps the base framework install small and moves concrete adapter stacks
behind explicit extras.

## Goal

Keep the core framework install small and predictable, while moving concrete
adapter stacks behind explicit extras.

## Target Dependency Groups

- `core`: required for pipeline construction, config parsing, runtime execution,
  events, artifacts, and plugin registration.
- `opencv`: OpenCV frame extractors, processors, trackers, ArUco, and video
  exporters.
- `visualization`: Matplotlib visualizers.
- `ui`: Streamlit Studio and UI services.
- `yolo`: Ultralytics pose/skeleton extractors.
- `pose`: COCO pose analyzer model helpers such as `joblib` and
  `scikit-learn`.
- `dev`: pytest, ruff, docs tooling, and local QA helpers.

## Implemented Refactor

- `pyproject.toml` defines minimal base dependencies plus `opencv`,
  `visualization`, `ui`, `yolo`, `pose`, `all`, and `dev` extras.
- Built-in registry creation uses lazy factories for optional adapter
  components, so listing built-ins does not import OpenCV, Matplotlib,
  Streamlit, or Ultralytics.
- Optional adapter creation raises actionable install guidance when a required
  package is missing.
- `sef doctor` reports missing adapter groups as warnings instead of core
  install failures.

## Install Examples

```bash
pip install -e .
pip install -e ".[opencv]"
pip install -e ".[visualization]"
pip install -e ".[ui]"
pip install -e ".[yolo]"
pip install -e ".[pose]"
pip install -e ".[all]"
pip install -e ".[dev]"
```

## Acceptance Criteria

- `import sef`, `import sef.core`, and `sef.pipeline(...)` work without OpenCV,
  Matplotlib, Streamlit, or Ultralytics installed.
- Built-in OpenCV components fail lazily with clear install guidance when the
  `opencv` extra is missing.
- UI imports are never required by the framework core or CLI validation path.
- `sef doctor` distinguishes core health from optional adapter availability.
- The full development environment still passes the same pipeline and adapter
  tests as before the split.
