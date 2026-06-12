# Core Pipeline Architecture

This folder contains the SEF pipeline application engine. The code here does
not implement OpenCV algorithms, Matplotlib visualizations, or UI logic:
it coordinates external components through interfaces, applies execution rules,
and produces reproducible outputs.

## Architectural Goals

- Keep the core independent from concrete components.
- Separate construction, planning, execution, monitoring, and outputs.
- Support batch, streaming, and hybrid pipelines without duplicating logic.
- Make failures diagnosable through explicit stages.
- Preserve reproducibility of every run through config and code export.
- Encourage extensibility through interfaces and small replaceable collaborators.

## Component Responsibilities

### Public Facade

- `Pipeline`: entry point for executing a single `PipelineContext`. It
  coordinates injector, executor, and output assembler, but contains no stage
  logic.
- `PipelineOrchestrator`: application facade for synchronous runs, asynchronous
  submissions, and event-driven triggers.
- `ThreadedPipelineRunner`: executes pipelines through `ThreadPoolExecutor`,
  handling lifecycle events, retries, ID deduplication, and monitoring.

### Construction and Configuration

- `PipelineContext`: immutable container for pipeline dependencies. It validates
  structural invariants before execution.
- `FluentPipelineBuilder`: programmatic builder for Python code.
- `ConfigPipelineBuilder`: declarative builder based on external configurations
  and `PluginRegistry`.
- `DefaultPipelineFactory`: default factory that builds a `Pipeline` from a
  context and runtime metadata.
- `StreamRuntimeConfig` and `LatencyPolicy`: configure buffers, backpressure,
  and latency policies for streaming.

### Planning

- `PipelineExecutionPlanner`: generates a readable execution plan before the run.
- `PipelineExecutionPlan`: represents stages, materialization boundaries,
  batch/streaming modes, and memory estimates.
- `PipelineComponentCapabilities`: single source of truth for determining
  whether a component can run in streaming mode.

### Execution

- `SegmentedPipelineExecutor`: traverses the entire pipeline and selects the
  best execution mode stage by stage by delegating to segment executors.
- `FrameSegmentExecutor`: manages frame extraction, frame processors, and frame
  exporters.
- `SignalSegmentExecutor`: manages signal extraction and signal cleaners.
- `AnalysisSegmentExecutor`: manages batch/streaming analyzers, fan-out, and
  final visualizers.
- `PipelineBoundaryMaterializer`: materializes frames or signals only at
  explicit boundaries between streaming segments and batch stages.
- `PipelineExecutionResources`: collects shared buffers and artifacts for a
  single run.
- `PipelineExecutionPolicy`: strategy contract for deciding batch vs streaming.
  `DefaultPipelineExecutionPolicy` is the default implementation, but it can be
  replaced with latency-first, memory-first, or domain-specific policies.
- `PipelineExecutionLookahead`: answers questions about downstream streamable
  stages, sharing the same logic between planner and runtime.
- `PipelineRuntimeState`: represents the current runtime state of frames and
  signals, distinguishing materialized data from streams with pending tasks.
- `VisualizationExecutor`: resolves visualizer bindings and creates the related
  `VisualizationContext`.
- `PipelineStageExecutor`: executes a single stage and normalizes errors into
  `PipelineExecutionError`.
- `PipelineBuffers`: utilities for buffer materialization and abort handling.

### Outputs, Export, and Observability

- `PipelineOutputAssembler`: converts the raw result into `PipelineOutputs`,
  attaching metadata, execution plans, and reproducibility artifacts.
- `PipelineConfigExporter`: exports a run into declarative configuration.
- `PipelineCodeExporter`: generates Python code equivalent to the configuration.
- `PipelineExportUtils`: helper functions for JSON/YAML serialization.
- `InMemoryPipelineMonitor`: in-memory monitor for pipeline run states.
- `InMemoryPipelineOutputStore`: optional in-memory store for outputs.
- `PipelineRunSnapshot`: immutable snapshot of observable state.

### Events and Branching

- `PipelineEventInjector`: injects event bus and metadata into components that
  implement `IEventEmitter`.
- `BranchingCoordinator`: listens to domain events, evaluates branching rules,
  and dispatches triggers for secondary pipelines.
- `VisualizerBinding`: connects visualizers to specific analyzer results.
  The binding validates and resolves target indices in a single place.
- `IntermediateFrameCapture`: captures intermediate snapshots for debugging and
  artifacts.
- `FrameProcessingStage`: adapter for context-aware frame processors.
- `SingleFrameProcessorAdapter`: adapts an `ISingleFrameProcessor` to the
  `IFrameBufferProcessor` contract and streaming model.

## Execution Flow

```text
Pipeline.run()
  -> PipelineEventInjector.inject(...)
  -> SegmentedPipelineExecutor.run()
     -> frame_extraction
     -> frame_processing[*]
     -> frame_exporters[*]
     -> signal_extraction
     -> signal_cleaners[*]
     -> analyzers[*] + visualizers[*]
     -> intermediate_frame_visualizers[*]
  -> PipelineOutputAssembler.build(...)