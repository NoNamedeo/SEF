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
    "id": "experiment-001",
    "metadata": {
        "owner": "lab",
    },
    "run": {
        "execution_plan": "summary",
        "reproducibility": True,
        "runtime": {
            "frame_buffer_size": 8,
            "signal_buffer_size": 8,
            "data_buffer_size": 8,
        },
    },
    "pipeline": {
        ...
    },
}
```

`schema_version` is optional for legacy configs. Missing version means `1.0`.
New tools should always write it explicitly.

`id` is the optional run identifier propagated into events, output metadata, and
artifacts. `metadata` is optional descriptive data copied into execution
metadata.

`run` is optional. Omit it for the lowest-overhead execution path. The current
schema accepts only the `run` section for execution controls.

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

Configs must use `frame_processors` for frame preprocessing.

Runtime settings belong in `run.runtime` because they describe how a run is
executed, not which components belong to the pipeline graph. `pipeline.runtime`
is rejected by the current schema.

## Run Section

```python
{
    "execution_plan": "summary",
    "reproducibility": True,
    "runtime": {
        "frame_buffer_size": 8,
        "signal_buffer_size": 8,
        "data_buffer_size": 8,
        "latency_policy": {
            "name": "blocking",
            "params": {},
        },
    },
}
```

The `run` section controls execution behavior and metadata attached to completed
outputs. It does not describe pipeline components.

Fields:

- `execution_plan`: `none`, `summary`, or `full`. `none` is the lightweight
  default and does not build an execution plan for output metadata. Boolean
  values are also accepted: `True` maps to `full`, `False` maps to `none`.
- `reproducibility`: when `True`, outputs include normalized config, JSON/YAML
  exports, and generated Python rebuild code.
- `runtime`: bounded-buffer and latency-policy settings used by adaptive
  streaming.

CLI `sef run --output` writes summaries, normalized config, and artifacts
without forcing execution-plan metadata. Use `--explain` when you also want the
CLI to print and persist `execution_plan.*` files.

## Orchestration Boundary

Run config describes one executable pipeline run: identity, metadata, execution
settings, and the pipeline graph. It intentionally does not configure in-process
event buses, Python lifecycle handlers, output-store objects, or custom runner
instances.

That boundary is deliberate. Reusable orchestration state remains Python/API
behavior. Use `pipeline.run()` or `sef.run(config)` for normal execution,
`sef.submit(...)` for background execution, and `sef.orchestrator()` when a
shared orchestrator needs lifecycle callbacks, active-id tracking, branching, or
custom runner integration.

If a branching workflow needs to expose child pipeline results, model that at
the application/output layer: record which child pipelines ran, which events
triggered them, and which upstream data contributed to the final result. Do not
put branching topology into the YAML schema yet.

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

Runtime config lives under `run.runtime` in new configs. Buffer sizes must be
positive integers.

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
