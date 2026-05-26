# Reference: Component Contracts

This page summarizes the public component interfaces. See
[Plugin Authoring](../plugin-authoring.md) for implementation guidance.

## IFrameExtractor

Official import: `from library.core.interfaces import IFrameExtractor`

```python
def extract(self) -> FrameBuffer
```

Returns a closed `FrameBuffer` containing `Frame` objects.

## ISingleFrameProcessor

Official import: `from library.core.interfaces import ISingleFrameProcessor`

```python
def process(self, frame: Frame) -> Frame
```

Transforms one frame without sequence-level context.

## IFrameBufferProcessor

Official import: `from library.core.interfaces import IFrameBufferProcessor`

```python
def process(self, buffer: FrameBuffer) -> FrameBuffer
```

Transforms a complete frame sequence.

## IFrameExporter

Official import: `from library.core.interfaces import IFrameExporter`

```python
def export(self, buffer: FrameBuffer, context: FrameExportContext) -> FrameExportResult
```

Returns forwarded frames plus export artifacts.

## ISignalExtractor

Official import: `from library.core.interfaces import ISignalExtractor`

```python
def extract(self, buffer: FrameBuffer) -> ISignal
```

Converts frames into signal samples.

## ISignalCleaner

Official import: `from library.core.interfaces import ISignalCleaner`

```python
def clean(self, signal: ISignal) -> ISignal
```

Transforms signal samples.

## IAnalyzer

Official import: `from library.core.interfaces import IAnalyzer`

```python
def analyze(self, signal: ISignal) -> IData
```

Converts a signal into analytical data.

## IVisualizer

Official import: `from library.core.interfaces import IVisualizer`

```python
def render(self, data: IData, context: VisualizationContext | None = None) -> tuple[VisualArtifact, ...]
```

Converts analytical data into UI-agnostic artifacts.

## Streaming Variants

Official import: `from library.core.interfaces import IStreamingAnalyzer`

Streaming variants publish into buffers through `*_into()` methods and must
close output buffers on completion.
