# Reference: Error Model

Official import:

```python
from sef.core import PipelineConfigurationError, PipelineExecutionError
```

## Base Classes

- `SEFError`
- `PipelineError`
- `PipelineConfigurationError`
- `PipelineExecutionError`

## Configuration Errors

- `ConfigSchemaError`
- `ConfigVersionError`
- `PipelineContextError`
- `PluginResolutionError`
- `PluginConstructionError`
- `InvalidPipelineTriggerEventError`
- `LatencyPolicyError`

## Registry Errors

- `PluginRegistryError`
- `InvalidPluginRegistrationError`
- `DuplicatePluginRegistrationError`

## Runtime Errors

- `StageExecutionError`
- `PipelineExecutionError`
- `StreamRuntimeError`
- `StreamAbortedError`

## Structured Fields

Configuration errors may expose:

- `path`
- `cause`
- `metadata`

Stage errors expose:

- `context`
- `stage`
- `stage_group`
- `component_name`
- `component_type`
- `pipeline_id`
- `cause`
