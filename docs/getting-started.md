# Getting Started

This guide builds and runs a complete pipeline without relying on OpenCV, YOLO,
or UI components. The example is intentionally small so plugin authors can focus
on core contracts.

## Install Context

Run examples from the repository root so `library` is importable:

```bash
python examples/minimal_pipeline.py
```

## Minimal Pipeline

```python
from library.core import ConfigPipelineBuilder, Pipeline, PluginRegistry
from library.core.plugins import PluginCategory

registry = PluginRegistry()
registry.register(PluginCategory.FRAME_EXTRACTOR, "frames", DemoFrameExtractor)
registry.register(PluginCategory.SIGNAL_EXTRACTOR, "signals", DemoSignalExtractor)
registry.register(PluginCategory.ANALYZER, "sample_count", SampleCountAnalyzer)
registry.register(PluginCategory.VISUALIZER, "summary", SummaryVisualizer)

config = {
    "schema_version": "1.0",
    "pipeline": {
        "frame_extractor": {"name": "frames", "params": {"frame_count": 3}},
        "signal_extractor": {"name": "signals"},
        "analyzers": [{"name": "sample_count"}],
        "visualizers": [{"name": "summary", "result_indices": [0]}],
    },
}

context = ConfigPipelineBuilder(registry).build_context(config)
outputs = Pipeline(context, pipeline_id="docs-quickstart").run()

print(outputs.results[0].y)
print(outputs.final_artifacts[0].content)
```

The complete runnable file is [examples/minimal_pipeline.py](../examples/minimal_pipeline.py).

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
from library.core import PipelineConfigurationError, PipelineExecutionError

try:
    context = ConfigPipelineBuilder(registry).build_context(config)
    outputs = Pipeline(context).run()
except PipelineConfigurationError as exc:
    print(exc.path, exc)
except PipelineExecutionError as exc:
    print(exc.stage, exc.cause)
```

Configuration errors usually indicate invalid config shape, unknown plugins, or
invalid plugin constructor parameters. Execution errors indicate a stage failed
after a valid context was built.
