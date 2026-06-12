# Dependency Management Plan

SEF currently installs every supported adapter dependency by default. That is
convenient during development, but it makes the base framework heavier than the
core architecture requires.

This plan is intentionally limited to dependency cleanup. It does not change the
current package metadata yet.

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
- `phase-mag`: external phase magnification integration. This extra should only
  install Python-side helpers; MATLAB/Octave release assets remain an external
  runtime prerequisite.
- `dev`: pytest, ruff, docs tooling, and local QA helpers.

## Execution Steps

1. Map every import in `sef.builtin` and `ui` to the dependency that provides it.
2. Split mandatory dependencies in `pyproject.toml` into a minimal base set and
   optional extras.
3. Add lazy import errors for optional adapters with actionable messages such as
   `Install SEF with the opencv extra to use opencv_buffered`.
4. Update `sef doctor` to report missing optional dependencies by feature group.
5. Add tests that import `sef`, `sef.core`, and `sef.api` in a minimal
   environment without OpenCV, Streamlit, Matplotlib, or Ultralytics.
6. Add one smoke test per extra group in the full development environment.
7. Update README and docs install examples after the tests prove the split.

## Acceptance Criteria

- `import sef`, `import sef.core`, and `sef.pipeline(...)` work without OpenCV,
  Matplotlib, Streamlit, or Ultralytics installed.
- Built-in OpenCV components fail lazily with clear install guidance when the
  `opencv` extra is missing.
- UI imports are never required by the framework core or CLI validation path.
- `sef doctor` distinguishes core health from optional adapter availability.
- The full development environment still passes the same pipeline and adapter
  tests as before the split.
