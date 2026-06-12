# Configuration

SEF supports declarative pipeline construction through a versioned mapping.
`ConfigPipelineBuilder` validates the config, resolves plugin factories through
`PluginRegistry`, and returns a `PipelineContext`.

Use the CLI to validate a config before running it:

```bash
sef validate pipeline.yaml
sef validate pipeline.yaml --strict
sef run pipeline.yaml --dry-run --explain
sef run pipeline.yaml --output outputs/run-001
sef config schema --format yaml
```

Unknown fields are warnings by default so exploratory configs remain easy to
iterate on. With `--strict`, unknown fields become blocking errors.

## Top-Level Schema

```python
{
    "schema_version": "1.0",
    "pipeline": {
        ...
    },
}
```

`schema_version` is optional for legacy configs. Missing version means `1.0`.
New tools should always write it explicitly.

## Pipeline Section

Required fields:

- `frame_extractor`: plugin entry.
- `signal_extractor`: plugin entry.
- `analyzers`: non-empty list of plugin entries.

Optional fields:

- `frame_processors`: list of plugin entries.
- `signal_cleaners`: list of plugin entries.
- `visualizers`: list of plugin entries.
- `intermediate_frames`: debug capture config.
- `runtime`: streaming runtime config.

Legacy configs that use `frame_cleaners` are normalized to `frame_processors`.
New configs should always write `frame_processors`.

## Plugin Entry

```python
{
    "name": "plugin_name",
    "params": {"constructor_arg": "value"}
}
```

`name` is required and must resolve in the expected plugin category. `params` is
optional and must be a mapping when present.

## Frame Processor Entry

```python
{
    "name": "opencv_gray",
    "processor_type": "single_frame",
    "params": {}
}
```

`processor_type` values:

- `single_frame`: factory must create `ISingleFrameProcessor`;
- `frame_buffer`: factory must create `IFrameBufferProcessor`.

Missing `processor_type` defaults to `single_frame`.

## Visualizer Result Binding

```python
{
    "name": "summary_visualizer",
    "params": {},
    "result_indices": [0]
}
```

When `result_indices` is present, the visualizer is bound only to selected
analyzer outputs. Without it, the visualizer is unbound and can be applied by
default visualization behavior.

## Runtime Section

```python
{
    "frame_buffer_size": 8,
    "signal_buffer_size": 8,
    "data_buffer_size": 8,
    "latency_policy": {
        "name": "drop_oldest",
        "params": {}
    }
}
```

Buffer sizes must be positive integers.

Supported latency policies:

- `blocking`
- `drop_newest`
- `drop_oldest`
- `adaptive_sampling`

`adaptive_sampling` params:

- `min_interval`
- `max_interval`
- `high_watermark`
- `low_watermark`

## Intermediate Frames

```python
{
    "enabled": True,
    "sampling_interval": 10,
    "max_stored_frames": 20,
    "export_directory": "artifacts/debug",
    "lazy_saving": True,
    "visualizers": [{"name": "intermediate_frames_grid"}]
}
```

Intermediate frame capture is for debugging frame processing stages. It is not
normal analyzer output and should be rendered by dedicated debug visualizers.

## Version Migration

Config migrations are defined by `PipelineConfigMigration` and executed by
`PipelineConfigVersionManager`. Migrations must not mutate the caller's mapping.

Unsupported versions raise `ConfigVersionError` with:

- `version`
- `supported_versions`
- `path`
- `metadata`
