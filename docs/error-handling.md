# Error Handling

SEF exposes a typed error model so UIs, services, and tests can handle failures
without parsing logs.

## Base Types

`SEFError` is the root public error.

`PipelineError` is the base for pipeline-related failures.

`PipelineConfigurationError` represents invalid configuration or construction
state and remains a `ValueError` for compatibility.

## Configuration Errors

- `ConfigSchemaError`: invalid shape or field type.
- `ConfigVersionError`: unsupported or malformed config version.
- `PipelineContextError`: invalid `PipelineContext` invariants.
- `PluginResolutionError`: unknown plugin in a config path.
- `PluginConstructionError`: plugin found but constructor failed.
- `LatencyPolicyError`: invalid runtime latency policy config.

Use `path` to highlight the config field in UI:

```python
try:
    context = ConfigPipelineBuilder(registry).build_context(config)
except PipelineConfigurationError as exc:
    show_config_error(path=exc.path, message=str(exc), metadata=exc.metadata)
```

## Execution Errors

`PipelineExecutionError` wraps failures raised by a stage and preserves:

- `stage`
- `stage_group`
- `component_name`
- `component_type`
- `pipeline_id`
- `cause`

```python
try:
    outputs = Pipeline(context).run()
except PipelineExecutionError as exc:
    logger.exception("Stage failed", extra=exc.context.as_dict())
    raise
```

## Registry Errors

- `InvalidPluginRegistrationError`
- `DuplicatePluginRegistrationError`

Registry lookup uses `KeyError`; builders convert lookup failures to
`PluginResolutionError` with category, name, and available plugin names.

## UI Mapping

Recommended UI severity:

- `ConfigSchemaError`: user-fixable form/config error.
- `ConfigVersionError`: migration or compatibility error.
- `PluginResolutionError`: missing plugin or wrong registry bootstrap.
- `PluginConstructionError`: invalid params or plugin initialization failure.
- `PipelineExecutionError`: runtime failure after launch.
- `StreamAbortedError`: cancellation or cooperative stream termination.
