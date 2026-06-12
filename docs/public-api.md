# Public API

The public API is the set of names exported from package initializers. External
code should import from these packages instead of relying on file layout.

## Recommended Imports

Use `sef` for normal application code:

```python
import sef

outputs = (
    sef.video("videos/Baloons.mp4", max_frames=300)
    .extract("opencv_tracker", tracker_type="MIL", start_box=[100, 200, 50, 80])
    .analyze("vertical_position")
    .run()
)
```

Use `sef.core` or `sef.core` only when you need lower-level contracts,
custom registries, or runtime integration:

```python
from sef.core import (
    ConfigPipelineBuilder,
    Pipeline,
    PipelineConfigurationError,
    PipelineExecutionError,
    PluginRegistry,
)
from sef.core.interfaces import IAnalyzer, IFrameExtractor, StageCapabilities
from sef.core.pipeline import CURRENT_PIPELINE_CONFIG_VERSION
from sef.core.plugins import PluginCategory
from sef.core.visualization import TextArtifact
```

## Public Packages

`sef` exposes high-level convenience imports.

`sef.core` exposes common core contracts and error types.

`sef.core.artifacts` exposes frame, signal, and analyzer data values.

`sef.core.events` exposes event contracts.

`sef.core.interfaces` exposes component interfaces.

`sef.core.interfaces.pipeline` exposes orchestration ports.

`sef.core.pipeline` exposes builders, runtime policies, execution plans, and
config versioning.

`sef.core.plugins` exposes registry contracts.

`sef.core.realtime` exposes realtime preview publication contracts.

`sef.core.visualization` exposes artifact and output contracts.

## Public CLI

The `sef` console script, and the equivalent `python -m sef`, expose the public
CLI surface:

```bash
sef init [tracking-demo] [--force]
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
