# Public API

The public API is the set of names exported from package initializers. External
code should import from these packages instead of relying on file layout.

## Recommended Imports

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
