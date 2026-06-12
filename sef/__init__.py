"""Public Pythonic entrypoint for SEF.

Use this module for concise workflows:

```python
import sef

outputs = (
    sef.pipeline("quickstart")
    .frames("demo_frames", frame_count=3)
    .signals("demo_signals")
    .analyze("sample_count")
    .visualize("summary_text")
    .run()
)
```

Advanced users can still import stable lower-level contracts from
``sef.core`` or ``sef.core``.
"""

from sef.api import (
    OrchestratorFacade,
    PipelineFacade,
    analyzer,
    cleaner,
    default_registry,
    frame_extractor,
    from_config,
    load_config,
    normalize_config,
    orchestrator,
    pipeline,
    processor,
    register_user_plugin,
    signal_extractor,
    video,
    visualizer,
    webcam,
)
from sef.core import PipelineExecutionError

__all__ = [
    "OrchestratorFacade",
    "PipelineFacade",
    "PipelineExecutionError",
    "analyzer",
    "cleaner",
    "default_registry",
    "frame_extractor",
    "from_config",
    "load_config",
    "normalize_config",
    "orchestrator",
    "pipeline",
    "processor",
    "register_user_plugin",
    "signal_extractor",
    "video",
    "visualizer",
    "webcam",
]
