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

Use `library.core` or `sef.core` only when you need lower-level contracts,
custom registries, or runtime integration:

```python
from library.core import (
    ConfigPipelineBuilder,
    Pipeline,
    PipelineConfigurationError,
    PipelineExecutionError,
    PluginRegistry,
)
from library.core.interfaces import IAnalyzer, IFrameExtractor, StageCapabilities
from library.core.pipeline import CURRENT_PIPELINE_CONFIG_VERSION
from library.core.plugins import PluginCategory
from library.core.visualization import TextArtifact
```

## Public Packages

`library` exposes high-level convenience imports.

`library.core` exposes common core contracts and error types.

`library.core.artifacts` exposes frame, signal, and analyzer data values.

`library.core.events` exposes event contracts.

`library.core.interfaces` exposes component interfaces.

`library.core.interfaces.pipeline` exposes orchestration ports.

`library.core.pipeline` exposes builders, runtime policies, execution plans, and
config versioning.

`library.core.plugins` exposes registry contracts.

`library.core.realtime` exposes realtime preview publication contracts.

`library.core.visualization` exposes artifact and output contracts.

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

Direct imports such as `library.core.pipeline.SomeInternalExecutor` may work but
are not automatically public. Use package exports for integration code and docs.
