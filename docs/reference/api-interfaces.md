# API: Interfaces Package

The interfaces package defines ports implemented by extractors, processors,
cleaners, analyzers, visualizers, buffers, and orchestration adapters.

## Stage Capabilities

::: library.core.interfaces.StageCapabilities.StageCapabilities

## Batch Component Contracts

::: library.core.interfaces.IFrameExtractor.IFrameExtractor

::: library.core.interfaces.ISingleFrameProcessor.ISingleFrameProcessor

::: library.core.interfaces.IFrameBufferProcessor.IFrameBufferProcessor

::: library.core.interfaces.IFrameExporter.FrameExportContext

::: library.core.interfaces.IFrameExporter.FrameExportResult

::: library.core.interfaces.IFrameExporter.IFrameExporter

::: library.core.interfaces.ISignalExtractor.ISignalExtractor

::: library.core.interfaces.ISignalCleaner.ISignalCleaner

::: library.core.interfaces.IAnalyzer.IAnalyzer

::: library.core.interfaces.IVisualizer.IVisualizer

## Data Contracts

::: library.core.interfaces.IData.IData

::: library.core.interfaces.ISignal.ISignal

::: library.core.interfaces.ISignalSample.ISignalSample

## Streaming Contracts

::: library.core.interfaces.BufferContracts.IBuffer

::: library.core.interfaces.BufferContracts.IAbortableBuffer

::: library.core.interfaces.BufferContracts.IBufferSubscription

::: library.core.interfaces.BufferContracts.ISubscribableBuffer

::: library.core.interfaces.BufferContracts.IFrameBuffer

::: library.core.interfaces.StreamingContracts.IStreamingFrameExtractor

::: library.core.interfaces.StreamingContracts.IStreamingFrameBufferProcessor

::: library.core.interfaces.StreamingContracts.IStreamingFrameExporter

::: library.core.interfaces.StreamingContracts.IStreamingSignalExtractor

::: library.core.interfaces.StreamingContracts.IStreamingSignalCleaner

::: library.core.interfaces.StreamingContracts.IStreamingAnalyzer

::: library.core.interfaces.StreamingContracts.IStreamingVisualizer

## Orchestration Ports

::: library.core.interfaces.pipeline.IPipelineFactory.IPipelineFactory

::: library.core.interfaces.pipeline.IPipelineRunner.IPipelineRunner

::: library.core.interfaces.pipeline.IPipelineMonitor.IPipelineMonitor

::: library.core.interfaces.pipeline.IPipelineOutputStore.IPipelineOutputStore

::: library.core.interfaces.pipeline.IPipelineValidator.IPipelineValidator

::: library.core.interfaces.pipeline.IEventBus.IEventBus

::: library.core.interfaces.pipeline.IRetryPolicy.IRetryPolicy

::: library.core.interfaces.pipeline.IBranchingRule.IBranchingRule

