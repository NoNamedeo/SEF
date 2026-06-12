# Reference: Generated API

This section is generated with `mkdocstrings` from the public contracts exported
by SEF package initializers.

Build locally with:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Generated pages are split by public package so readers can navigate by
integration concern instead of internal file layout.

- [Core Package](api-core.md)
- [Pipeline Package](api-pipeline.md)
- [Interfaces Package](api-interfaces.md)
- [Artifacts Package](api-artifacts.md)
- [Plugins Package](api-plugins.md)
- [Visualization Package](api-visualization.md)
- [Events Package](api-events.md)
- [Realtime Package](api-realtime.md)

Public imports should use package-level exports, for example:

```python
from sef.core import ConfigPipelineBuilder, Pipeline
from sef.core.interfaces import IAnalyzer, IStreamingAnalyzer
from sef.core.plugins import PluginCategory, PluginRegistry
```
