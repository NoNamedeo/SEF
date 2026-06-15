# Plugin Authoring

Plugins are concrete implementations of public interfaces. A plugin should do
one job, declare capabilities accurately, and return public data contracts.

## Authoring Checklist

1. Choose the narrowest interface.
2. Keep constructor parameters stable and explicit.
3. Accept `config: dict | None = None` for optional structured config.
4. Set `capabilities` conservatively.
5. Preserve frame indexes, timestamps, and metadata when meaningful.
6. Close or abort streaming output buffers correctly.
7. Return `IData` from analyzers and `VisualArtifact` from visualizers.
8. Register with category, name, description, version, aliases, and metadata.
9. Add one config-builder test and one execution test.
10. Document parameters and output shape.

## Analyzer Example

```python

from sef.core.artifacts.data import TwoDimGraphData
from sef.core.interfaces import IAnalyzer, ISignal


class SampleCountAnalyzer(IAnalyzer):
    def analyze(self, signal: ISignal) -> TwoDimGraphData:
        count = len(list(signal))
        return TwoDimGraphData(x=[0.0], y=[float(count)], title="Sample count")
```

## Function Plugin Example

Use decorators for small components whose behavior fits a single function.
Decorator metadata is public: it appears in `sef components inspect`, JSON
catalog output, and registry descriptors.

```python
import sef
from sef.core.artifacts.data import TwoDimGraphData
from sef.core.interfaces import StageCapabilities


@sef.analyzer(
    "vertical_velocity",
    description="Estimate vertical velocity from a tracked signal.",
    version="1.0.0",
    aliases=("velocity_y",),
    metadata={
        "domain": "motion",
        "tags": ["tracking", "kinematics"],
        "output": "TwoDimGraphData",
        "params": {
            "fps": {"type": "float", "default": 30.0, "min": 0.0},
        },
    },
    capabilities=StageCapabilities.streaming(stateful=False, realtime_safe=True),
)
def vertical_velocity(signal, fps: float = 30.0):
    samples = list(signal)
    return TwoDimGraphData(x=[0.0], y=[float(len(samples)) * fps], title="Velocity")
```

Recommended metadata keys are documented in [Plugin Metadata](plugin-metadata.md).
The registry does not enforce that shape; keep metadata JSON-friendly so CLI
tools and UI catalogs can render it without importing the component.

Use `@sef.frame_buffer_processor` when the function needs the whole frame
sequence instead of one frame at a time:

```python
@sef.frame_buffer_processor(
    "temporal_filter",
    description="Apply a temporal filter to the complete frame sequence.",
)
def temporal_filter(buffer, strength: float = 0.5):
    return buffer
```

## Visualizer Example

```python
from sef.core.interfaces import IData, IVisualizer
from sef.core.visualization import TextArtifact, VisualizationContext


class SummaryVisualizer(IVisualizer):
    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[TextArtifact, ...]:
        return (
            TextArtifact(
                kind="text",
                title="Summary",
                content=f"Rendered by {context.visualizer_name if context else 'unknown'}",
            ),
        )
```

## Capability Declaration

```python
from sef.core.interfaces import StageCapabilities


class RealtimeSafeProcessor:
    capabilities = StageCapabilities.streaming(
        stateful=False,
        preserves_order=True,
        realtime_safe=True,
    )
```

Do not declare streaming support unless the implementation can publish output
progressively without requiring the full upstream sequence.

## Anti-Patterns

Avoid importing Streamlit, web frameworks, or application state in plugins.

Avoid returning raw UI objects. Return `VisualArtifact`.

Avoid broad `**kwargs` constructors for public plugins; they make config
validation and reproducibility weaker.

Avoid mutating shared global state during execution.

Avoid swallowing exceptions that should become typed pipeline failures.
