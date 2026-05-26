# Streaming Runtime

The SEF runtime can execute compatible stages as bounded streams. Streaming is
not an implementation detail; it is a declared contract through interfaces and
`StageCapabilities`.

## Streaming Interfaces

- `IStreamingFrameExtractor`
- `IStreamingFrameBufferProcessor`
- `IStreamingFrameExporter`
- `IStreamingSignalExtractor`
- `IStreamingSignalCleaner`
- `IStreamingAnalyzer`
- `IStreamingVisualizer`

Batch interfaces remain valid. The planner decides whether a stage runs in
batch or streaming mode based on capabilities, upstream state, and downstream
demand.

## Execution Policy

`PipelineExecutionPolicy` decides execution mode for each stage. The default
policy:

- keeps an active stream when a stage can preserve it;
- avoids isolated streaming switches when no downstream stage benefits;
- opens streaming segments for progressive consumers;
- uses bounded queue estimates when available.

Custom policies can implement memory-first, latency-first, or domain-specific
decisions.

## Buffers

Frame streams use `IFrameBuffer`, which supports:

- blocking `put()`;
- non-blocking `try_put()`;
- `drop_oldest()`;
- `fill_ratio()`;
- cooperative `abort()`.

Signal and data streams use generic buffer contracts with explicit consumer
counts where fan-out is required.

## Latency Policies

`blocking` preserves every accepted frame and may increase latency.

`drop_newest` rejects new frames when the queue is full.

`drop_oldest` favors recent frames by discarding older queued frames.

`adaptive_sampling` increases sampling interval under pressure.

Use `drop_oldest` for realtime previews where freshness matters more than
perfect frame coverage. Use `blocking` for offline reproducibility.

## Streaming Component Rules

Streaming components must:

- publish in input order unless documented otherwise;
- close output buffers on normal completion;
- abort cooperatively on downstream cancellation;
- avoid reading the entire input before publishing first output;
- avoid UI side effects in core processing paths;
- keep per-run mutable state inside the component instance or method scope.

## Realtime Visualization

Realtime visualizers should consume progressive analyzer data and publish
UI-agnostic realtime frames or artifacts through adapter-safe contracts. A
visualizer should not assume OpenCV GUI windows or Streamlit containers are
available unless it is explicitly an adapter outside core.
