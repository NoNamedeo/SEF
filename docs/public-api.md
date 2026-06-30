# Public API

The public API is the set of names exported from package initializers. External
code should import from these packages instead of relying on file layout.

## Recommended Imports

Use `sef` for normal application code. The public mental model is deliberately
small:

- `sef.pipeline(...)`, `sef.video(...)`, or `sef.webcam(...)` describe one pipeline.
- `.run()` on that builder is the fluent single-pipeline execution path.
- `sef.run(config_or_pipeline)` and `sef.submit(config_or_pipeline)` execute
  existing run configs or pipeline builders through the same runtime.
- `sef.orchestrator()` coordinates execution when lifecycle events, background
  submission, branching, shared active-id state, or custom runtime wiring are
  needed.
- `sef.from_config(...)` loads the same pipeline model from YAML/JSON data.
- Decorators such as `@sef.analyzer(...)` and
  `@sef.frame_buffer_processor(...)` register function plugins with optional
  description, version, aliases, metadata, and capabilities.

The builder object returned by `sef.pipeline(...)` is named `PipelineFacade` in
type hints, but users should treat it as the fluent pipeline builder rather than
as a separate runtime concept.

```python
import sef

outputs = (
    sef.video("videos/Baloons.mp4", max_frames=300)
    .extract("opencv_tracker", tracker_type="MIL", start_box=[100, 200, 50, 80])
    .analyze("vertical_position")
    .run(id="tracked-run", metadata={"owner": "lab"})
)
```

The same runtime also accepts a run config:

```python
config = (
    sef.video("videos/Baloons.mp4", max_frames=300)
    .extract("opencv_tracker", tracker_type="MIL", start_box=[100, 200, 50, 80])
    .analyze("vertical_position")
    .to_config(id="tracked-run", metadata={"owner": "lab"})
)

outputs = sef.run(config)
```

Use `sef.orchestrator()` when execution needs reusable lifecycle observation,
background submission, shared runner state, or simple event-driven branching:

```python
import sef

events = []
pipeline = (
    sef.pipeline("tracked-run")
    .frames(MyFrameExtractor)
    .signals(MySignalExtractor)
    .analyze(MyAnalyzer)
)

outputs = (
    sef.orchestrator()
    .on_lifecycle("after_run", events.append)
    .run(pipeline)
)
```

All public run paths converge on the same orchestrator runtime:

- `pipeline.run()` is fluent convenience for one immediate pipeline run.
- `sef.run(config)` executes an already-built run config.
- `sef.orchestrator().run(...)` uses a reusable orchestrator instance. Use it
  when the run needs lifecycle callbacks, async submission through `submit()`,
  active-id tracking, branching, or custom runner integration.

Use `sef.core` only when you need lower-level contracts, custom registries, or
runtime integration:

```python
from sef.core import (
    ConfigPipelineBuilder,
    EventBus,
    IBranchingRule,
    Pipeline,
    PipelineOrchestrator,
    PipelineConfigurationError,
    PipelineExecutionError,
    PluginRegistry,
)
from sef.core.interfaces import IAnalyzer, IFrameExtractor, StageCapabilities
from sef.core.pipeline import CURRENT_PIPELINE_CONFIG_VERSION
from sef.core.plugins import PluginCategory
from sef.core.visualization import TextArtifact
from sef.builtin.registry import create_builtin_registry
```

## Public Packages

`sef` exposes high-level convenience imports.

`sef.orchestrator()` exposes the simple orchestration path for synchronous runs,
background runs, lifecycle callbacks, and branching rules.

`sef.core` exposes common core contracts and error types.

`sef.core.artifacts` exposes frame, signal, and analyzer data values.

`sef.core.events` exposes event contracts.

`sef.core.interfaces` exposes component interfaces.

`sef.core.interfaces.pipeline` exposes orchestration ports.

`sef.core.pipeline` exposes builders, runtime policies, execution plans, and
config versioning.

`sef.core.plugins` exposes registry contracts.

`sef.builtin.registry` exposes the built-in registry adapter for OpenCV,
Matplotlib, and other concrete SEF components. It intentionally lives outside
`sef.core` so the core registry stays independent from concrete adapters.

`sef.core.realtime` exposes realtime preview publication contracts.

`sef.core.visualization` exposes artifact and output contracts.

## Public CLI

The `sef` console script, and the equivalent `python -m sef`, expose the public
CLI surface:

```bash
sef init [tracking-demo|plugin] [--force]
sef doctor [--config pipeline.yaml]
sef validate <config> [--strict] [--debug]
sef run <config> [--dry-run] [--explain] [--output outputs/run] [--debug]
sef components list [--category analyzer] [--json]
sef components inspect <name> [--category analyzer] [--json]
sef config schema [--format json|yaml]
sef version
```

CLI commands load builtin components plus local plugin modules from `plugins/*.py`.
Errors are rendered as readable messages by default; pass `--debug` to include a
full traceback for `run` and `validate` failures.

The CLI config schema is intentionally pipeline-only. It does not accept
branching or orchestration fields. Use Python orchestration APIs for those
workflows.

## Compatibility Rules

Compatible changes:

- adding optional metadata fields;
- adding new public exports;
- adding new plugin descriptor fields;
- adding new config migrations;
- adding subclasses of existing public errors.

Potentially breaking changes:

- removing or renaming public exports;
- changing an abstract method signature;
- changing config semantics without a migration;
- changing error inheritance;
- changing `PipelineOutputs` field meaning.

## Internal Implementation Warning

Direct imports such as `sef.core.pipeline.SomeInternalExecutor` may work but
are not automatically public. Use package exports for integration code and docs.
