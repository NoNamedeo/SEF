# Reference: Data and Artifacts

## Frame

Official import: `from sef.core.artifacts import Frame`

Fields:

- `image`
- `index`
- `timestamp_seconds`
- `metadata`

`frame.frame` is a compatibility alias for `frame.image`.

## FrameBuffer

Official import: `from sef.core.artifacts import FrameBuffer`

Core methods:

- `put(frame)`
- `try_put(frame)`
- `drop_oldest()`
- `close()`
- `abort()`
- `fill_ratio()`
- iteration over frames

## ISignal and ISignalSample

Official import:

```python
from sef.core.interfaces import ISignal, ISignalSample
```

Signals are iterable collections of samples. Samples expose frame index,
optional timestamp, and metadata.

## IData

Official import: `from sef.core.interfaces import IData`

Analyzer outputs should implement `IData`. Built-in data values include graph
data, point data, trajectory data, pose data, tracking data, mask artifacts, and
domain-specific analysis records.

## VisualArtifact

Official import: `from sef.core.visualization import VisualArtifact`

Common fields:

- `artifact_id`
- `kind`
- `role`
- `title`
- `description`
- `metadata`

Concrete artifact types:

- `ImageArtifact`
- `VideoArtifact`
- `VideoFileArtifact`
- `DeferredVideoArtifact`
- `TableArtifact`
- `JsonArtifact`
- `TextArtifact`

## PipelineOutputs

Official import: `from sef.core.visualization import PipelineOutputs`

Fields:

- `results`
- `final_artifacts`
- `debug_artifacts`
- `metadata`
- `intermediate_frames`

`artifact_count` returns final plus debug artifact count.
