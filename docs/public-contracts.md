# SEF Public Contracts

This page is the compact single-file contract reference. The recommended
documentation entry point is now [docs/index.md](index.md), which splits the
same contract surface into focused guides, reference pages, and runnable
examples.

This document defines the public contracts for the SEF core package. It is
intended for plugin authors, UI adapters, service integrations, and maintainers
who need stable extension points without depending on implementation internals.

The contracts described here cover the Python package currently exposed by
`sef.core` and its public subpackages. Concrete OpenCV, YOLO, Matplotlib,
Streamlit, and UI modules are adapters built on top of these contracts.

## Stability Policy

Public contracts are importable from one of these package entry points:

- `sef`
- `sef.core`
- `sef.core.artifacts`
- `sef.core.events`
- `sef.core.interfaces`
- `sef.core.interfaces.pipeline`
- `sef.core.pipeline`
- `sef.core.plugins`
- `sef.core.realtime`
- `sef.core.visualization`

Implementation modules under `sef.core.pipeline` are not automatically
public only because they are importable by path. Prefer the package-level
exports above when building external code. A symbol should be considered stable
when it is exported through `__all__` from a public package initializer.

The following changes require a minor or major version bump:

- adding a required abstract method to a public interface;
- changing a method return type or side-effect contract;
- changing config schema semantics;
- removing or renaming a public export;
- changing public error inheritance.

The following changes are compatible:

- adding optional fields to metadata objects;
- adding new plugin categories or registry descriptor fields;
- adding new config schema versions with migrations;
- adding new error subclasses that still inherit from the documented base class.

## Public API Imports

Use package-level imports for application code:

```python
from sef.core import ConfigPipelineBuilder, Pipeline, PluginRegistry
from sef.core.interfaces import IAnalyzer, IFrameExtractor
from sef.core.pipeline import CURRENT_PIPELINE_CONFIG_VERSION
from sef.core.visualization import TextArtifact
```

Direct file-level imports are acceptable inside the SEF codebase and tests, but
external integrations should avoid relying on internal file layout.

## Configuration Contract

Declarative pipeline configuration is a mapping with a top-level `pipeline`
section and an optional top-level `schema_version`.

```python
from sef.core.pipeline import CURRENT_PIPELINE_CONFIG_VERSION

config = {
    "schema_version": CURRENT_PIPELINE_CONFIG_VERSION,
    "run": {
        "runtime": {
            "frame_buffer_size": 8,
            "signal_buffer_size": 8,
            "data_buffer_size": 8,
            "latency_policy": {"name": "blocking", "params": {}},
        },
    },
    "pipeline": {
        "frame_extractor": {"name": "opencv_buffered", "params": {"path": "video.mp4"}},
        "frame_processors": [],
        "signal_extractor": {"name": "opencv_tracker", "params": {}},
        "signal_cleaners": [],
        "analyzers": [{"name": "vertical_position"}],
        "visualizers": [{"name": "matplotlib", "result_indices": [0]}],
    },
}
```

Current public schema version: `1.0`.

Unversioned configs are treated as legacy-compatible `1.0` configs. New code
should emit `schema_version` explicitly.

`ConfigPipelineBuilder` validates and normalizes configs through
`PipelineConfigVersionManager` before constructing `PipelineContext`.
Unsupported versions raise `ConfigVersionError`.

### Versioning Rules

Schema migrations belong in `PipelineConfigVersioning.py`, not in
`ConfigPipelineBuilder`. A migration must be pure: it receives a mapping and
returns a new mapping. The builder should consume only the normalized current
schema.

Future schema versions should follow this process:

1. Add a `PipelineConfigMigration`.
2. Extend `SUPPORTED_PIPELINE_CONFIG_VERSIONS` when the new version becomes
   public.
3. Add tests for old config input, migrated output, and unsupported versions.
4. Keep exported configs on the current schema version.

## Plugin Registry Contract

`PluginRegistry` is the public catalog for pipeline component factories.
Builders resolve declarative config entries by plugin category and plugin name.

```python
from sef.core.plugins import PluginCategory, PluginRegistry

registry = PluginRegistry()
registry.register(
    PluginCategory.ANALYZER,
    "my_analyzer",
    MyAnalyzer,
    "Compute domain-specific metrics.",
    version="1.0.0",
    aliases=("my_analyzer_v1",),
    metadata={"owner": "analytics"},
)
```

### PluginDefinition

Every registered plugin is represented by an immutable `PluginDefinition`:

- `category`: canonical category string;
- `name`: canonical plugin name;
- `factory`: callable used to construct the component;
- `description`: human-readable description for UIs and diagnostics;
- `version`: plugin implementation version;
- `aliases`: additional names accepted by lookup and config resolution;
- `metadata`: immutable mapping for UI, ownership, tags, or compatibility data.

`PluginDefinition.as_dict()` returns a JSON-friendly descriptor without exposing
the factory object.

### Registry Guarantees

`PluginRegistry` guarantees:

- category/name/alias validation;
- duplicate rejection for canonical names and aliases;
- alias resolution through `get()` and `create()`;
- deterministic `available_names()` ordering;
- immutable snapshots through `snapshot()`;
- JSON-safe descriptors through `describe()`;
- read/write protection through an internal re-entrant lock.

Duplicate registrations raise `DuplicatePluginRegistrationError`.
Malformed registrations raise `InvalidPluginRegistrationError`.

### Canonical Categories

Public categories are exposed by `PluginCategory`:

- `frame_extractor`
- `single_frame_processor`
- `frame_buffer_processor`
- `signal_extractor`
- `signal_cleaner`
- `analyzer`
- `visualizer`
- `branching_rule`

Config entries must use a name registered in the expected category. The builder
does not search across categories.

## Pipeline Context Contract

`PipelineContext` is the immutable dependency holder for one executable
pipeline. It owns construction invariants, not execution behavior.

Required fields:

- `frame_extractor`
- `signal_extractor`
- `analyzers`, with at least one analyzer

Optional fields:

- `frame_processors`
- `frame_exporters`
- `signal_cleaners`
- `visualizers`
- `visualizer_bindings`
- `intermediate_frame_capture`
- `intermediate_frame_visualizers`
- `stream_runtime`
- `source_config`

The context normalizes component collections to tuples and rejects `None`
entries. Invalid context construction raises `PipelineContextError`, which is a
`PipelineConfigurationError` and remains a `ValueError` for compatibility.

The pipeline runtime assumes it receives a valid context and does not revalidate
component topology.

## Component Contracts

All components may expose a `config` dictionary. Constructors should accept
`config: dict | None = None` when component-level configuration is needed.
Components should avoid hidden global state and should report capabilities
conservatively.

### Frame Extractors

Interface: `IFrameExtractor`

```python
class IFrameExtractor:
    capabilities = StageCapabilities.batch()

    def extract(self) -> FrameBuffer:
        ...
```

A frame extractor is the source stage. It returns a `FrameBuffer` containing
`Frame` objects and must close the buffer when extraction is complete.

Streaming sources may implement `IStreamingFrameExtractor.extract_into()`.

### Frame Processors

Single-frame processors implement `ISingleFrameProcessor`:

```python
def process(self, frame: Frame) -> Frame:
    ...
```

Sequence-level processors implement `IFrameBufferProcessor`:

```python
def process(self, buffer: FrameBuffer) -> FrameBuffer:
    ...
```

Use `ISingleFrameProcessor` for stateless or per-frame transformations. Use
`IFrameBufferProcessor` when the algorithm requires temporal context,
multi-frame state, or whole-buffer operations.

Streaming processors may implement `IStreamingFrameBufferProcessor`.

### Frame Exporters

Interface: `IFrameExporter`

```python
def export(self, buffer: FrameBuffer, context: FrameExportContext) -> FrameExportResult:
    ...
```

Frame exporters persist or create artifacts from processed frames without being
frame processors. They must return a `FrameExportResult` containing:

- a `FrameBuffer` that downstream signal extractors can still consume;
- zero or more `VisualArtifact` instances.

Streaming exporters may implement `IStreamingFrameExporter.export_into()` and
must forward frames to the provided output buffer.

### Signal Extractors

Interface: `ISignalExtractor`

```python
def extract(self, buffer: FrameBuffer) -> ISignal:
    ...
```

Signal extractors convert frames into domain signal samples. The returned
`ISignal` must be iterable and should preserve frame order unless the component
explicitly documents otherwise.

Streaming signal extractors may implement `IStreamingSignalExtractor`.

### Signal Cleaners

Interface: `ISignalCleaner`

```python
def clean(self, signal: ISignal) -> ISignal:
    ...
```

Signal cleaners transform extracted signals without changing the pipeline
topology. They should preserve metadata and sample ordering when possible.

Streaming cleaners may implement `IStreamingSignalCleaner`.

### Analyzers

Interface: `IAnalyzer`

```python
def analyze(self, signal: ISignal) -> IData:
    ...
```

Analyzers convert a signal into analytical data. Results must implement `IData`
or use an existing data artifact class from `sef.core.artifacts`.

Streaming analyzers may implement `IStreamingAnalyzer.analyze_into()`. A
streaming analyzer may publish progressive `IData` values and must still return
a final `IData` result.

### Visualizers

Interface: `IVisualizer`

```python
def render(
    self,
    data: IData,
    context: VisualizationContext | None = None,
) -> tuple[VisualArtifact, ...]:
    ...
```

Visualizers convert analytical data into UI-agnostic artifacts. They must not
depend on Streamlit, browser state, or terminal display. UI-specific rendering
belongs in adapters that consume `VisualArtifact`.

Streaming visualizers may implement `IStreamingVisualizer.render_stream()` and
consume progressive analyzer data.

## Streaming and Buffer Contracts

Streaming support is opt-in. A component is considered stream-capable only when
it implements the appropriate streaming interface and declares compatible
`StageCapabilities`.

Public buffer protocols:

- `IBuffer[T]`: producer-side `put()` and `close()`;
- `IAbortableBuffer[T]`: cooperative `abort()`;
- `IBufferSubscription[T]`: consumer-side iterator with `abort()`;
- `ISubscribableBuffer[T]`: multi-consumer fan-out;
- `IFrameBuffer`: frame-specific queue with `capacity`, `try_put()`,
  `drop_oldest()`, and `fill_ratio()`.

Streaming implementations must:

- close output buffers on normal completion;
- abort upstream or downstream on unrecoverable failure where appropriate;
- avoid materializing full sequences unless their contract requires it;
- not publish after close or abort;
- preserve order unless `StageCapabilities.preserves_order` is false.

## Stage Capabilities

Every stage may expose a `capabilities` class or instance attribute with a
`StageCapabilities` value.

```python
from sef.core.interfaces import StageCapabilities

class MyStreamingAnalyzer(IStreamingAnalyzer):
    capabilities = StageCapabilities.streaming(stateful=True, realtime_safe=True)
```

Capability fields:

- `supports_streaming`
- `requires_complete_sequence`
- `stateful`
- `preserves_order`
- `supports_frame_parallelism`
- `realtime_safe`

The execution planner uses capabilities to decide batch vs streaming execution.
Declare capabilities conservatively. Never mark a component as streaming if it
must see the complete input before producing any output.

## Runtime and Latency Policy Contract

`StreamRuntimeConfig` controls bounded stream buffers:

- `frame_buffer_size`
- `signal_buffer_size`
- `data_buffer_size`
- `latency_policy`

Supported latency policies:

- `blocking`: preserve all frames and let the producer block;
- `drop_newest`: reject incoming frames when the frame queue is full;
- `drop_oldest`: discard queued frames to keep recent input;
- `adaptive_sampling`: increase sampling interval under queue pressure.

Invalid runtime or latency configuration raises `ConfigSchemaError` or
`LatencyPolicyError`.

## Artifacts and Data Contracts

### Frame

`Frame` is the public video frame value:

- `image`: NumPy array;
- `index`: optional frame index;
- `timestamp_seconds`: optional timestamp;
- `metadata`: mutable dictionary for component metadata.

`frame.frame` is a backward-compatible alias for `frame.image`.

### Signal and Samples

`ISignal` is iterable and contains `ISignalSample` instances. Concrete samples
must expose:

- `frame_index`;
- `timestamp_seconds`;
- `metadata`.

Domain-specific samples should subclass or follow `ISignalSample`.

### IData

`IData` is a marker base class for analyzer output. Public built-in result types
include graph data, point data, trajectory data, pose data, mask artifacts, and
domain-specific ArUco/COCO pose structures.

## Visualization Contract

`VisualArtifact` is the UI-agnostic presentation contract. Supported public
artifact types include:

- `ImageArtifact`
- `VideoArtifact`
- `VideoFileArtifact`
- `DeferredVideoArtifact`
- `TableArtifact`
- `JsonArtifact`
- `TextArtifact`

Artifacts include:

- `artifact_id`
- `kind`
- `role`
- `title`
- `description`
- `metadata`

Large videos should use `VideoFileArtifact` or `DeferredVideoArtifact` instead
of in-memory `VideoArtifact`.

`PipelineOutputs` is the final result aggregate:

- `results`: analyzer outputs;
- `final_artifacts`: user-facing artifacts;
- `debug_artifacts`: diagnostic artifacts;
- `metadata`: run metadata and reproducibility data;
- `intermediate_frames`: captured frame-processing debug snapshots.

## Events and Branching Contracts

`Event` is the public immutable event object. It carries:

- `event_type`
- `source`
- `payload`
- `correlation_id`
- `timestamp`
- `event_id`
- `metadata`

Components that emit events should implement `IEventEmitter`. The pipeline
injects the active event bus and execution metadata before running.

Branching rules implement `IBranchingRule`:

```python
def matches(self, event: Event) -> bool:
    ...

def build_config(self, event: Event) -> Mapping[str, Any]:
    ...
```

Branching rules should return a run config, be deterministic, and avoid mutating
the triggering event.

## Error Model

All public SEF errors inherit from `SEFError`.

Pipeline-level errors inherit from `PipelineError`.

Configuration errors inherit from `PipelineConfigurationError`, which also
inherits from `ValueError` for compatibility:

- `ConfigSchemaError`
- `ConfigVersionError`
- `PipelineContextError`
- `PluginResolutionError`
- `PluginConstructionError`
- `InvalidPipelineTriggerEventError`
- `LatencyPolicyError`

Registry errors:

- `PluginRegistryError`
- `InvalidPluginRegistrationError`
- `DuplicatePluginRegistrationError`

Execution errors:

- `StageExecutionError`
- `PipelineExecutionError`
- `StreamRuntimeError`
- `StreamAbortedError`

`StageExecutionError` and `PipelineExecutionError` preserve:

- `stage`
- `stage_group`
- `component_name`
- `component_type`
- `pipeline_id`
- `cause`

UI and service layers should inspect these structured fields instead of parsing
log messages.

## Plugin Authoring Checklist

Before publishing a new plugin:

1. Implement the narrowest applicable interface.
2. Set `capabilities` conservatively.
3. Accept stable constructor parameters and optional `config`.
4. Avoid importing UI frameworks from core/plugin logic.
5. Preserve frame indexes, timestamps, and metadata where meaningful.
6. Close or abort buffers correctly in streaming implementations.
7. Return UI-agnostic `VisualArtifact` objects from visualizers.
8. Register the plugin with category, name, description, version, and metadata.
9. Add config builder tests and at least one execution test.
10. Document plugin-specific params and output data shape.

## Minimal Plugin Example

```python

from sef.core.artifacts.data import TwoDimGraphData
from sef.core.interfaces import IAnalyzer, ISignal
from sef.core.plugins import PluginCategory, PluginRegistry


class SampleCountAnalyzer(IAnalyzer):
    """Return a one-point graph containing the number of signal samples."""

    def analyze(self, signal: ISignal) -> TwoDimGraphData:
        count = len(list(signal))
        return TwoDimGraphData(
            x=[0.0],
            y=[float(count)],
            label="samples",
            title="Sample count",
        )


registry = PluginRegistry()
registry.register(
    PluginCategory.ANALYZER,
    "sample_count",
    SampleCountAnalyzer,
    "Count signal samples.",
    version="1.0.0",
)
```

## Integration Boundary

The core package does not own:

- web permissions;
- Streamlit reruns;
- OpenCV window lifecycle;
- model downloads;
- device selection UI;
- long-term artifact persistence.

Those concerns belong to adapters or applications. The core contract is to
produce typed data, typed artifacts, typed errors, and observable execution
metadata through stable interfaces.
