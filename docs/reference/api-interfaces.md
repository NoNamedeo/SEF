# API: Interfaces Package

The interfaces package defines ports implemented by extractors, processors,
cleaners, analyzers, visualizers, buffers, and orchestration adapters.

## Stage Capabilities

::: sef.core.interfaces.StageCapabilities.StageCapabilities

## Batch Component Contracts

::: sef.core.interfaces.IFrameExtractor.IFrameExtractor

::: sef.core.interfaces.ISingleFrameProcessor.ISingleFrameProcessor

::: sef.core.interfaces.IFrameBufferProcessor.IFrameBufferProcessor

::: sef.core.interfaces.IFrameExporter.FrameExportContext

::: sef.core.interfaces.IFrameExporter.FrameExportResult

::: sef.core.interfaces.IFrameExporter.IFrameExporter

::: sef.core.interfaces.ISignalExtractor.ISignalExtractor

::: sef.core.interfaces.ISignalCleaner.ISignalCleaner

::: sef.core.interfaces.IAnalyzer.IAnalyzer

::: sef.core.interfaces.IVisualizer.IVisualizer

## Data Contracts

::: sef.core.interfaces.IData.IData

::: sef.core.interfaces.ISignal.ISignal

::: sef.core.interfaces.ISignalSample.ISignalSample

## Streaming Contracts

::: sef.core.interfaces.BufferContracts.IBuffer

::: sef.core.interfaces.BufferContracts.IAbortableBuffer

::: sef.core.interfaces.BufferContracts.IBufferSubscription

::: sef.core.interfaces.BufferContracts.ISubscribableBuffer

::: sef.core.interfaces.BufferContracts.IFrameBuffer

::: sef.core.interfaces.StreamingContracts.IStreamingFrameExtractor

::: sef.core.interfaces.StreamingContracts.IStreamingFrameBufferProcessor

::: sef.core.interfaces.StreamingContracts.IStreamingFrameExporter

::: sef.core.interfaces.StreamingContracts.IStreamingSignalExtractor

::: sef.core.interfaces.StreamingContracts.IStreamingSignalCleaner

::: sef.core.interfaces.StreamingContracts.IStreamingAnalyzer

::: sef.core.interfaces.StreamingContracts.IStreamingVisualizer

## Orchestration Ports

::: sef.core.interfaces.pipeline.IPipelineFactory.IPipelineFactory

::: sef.core.interfaces.pipeline.IPipelineRunner.IPipelineRunner

::: sef.core.interfaces.pipeline.IPipelineMonitor.IPipelineMonitor

::: sef.core.interfaces.pipeline.IPipelineOutputStore.IPipelineOutputStore

::: sef.core.interfaces.pipeline.IPipelineValidator.IPipelineValidator

::: sef.core.interfaces.pipeline.IEventBus.IEventBus

::: sef.core.interfaces.pipeline.IRetryPolicy.IRetryPolicy

::: sef.core.interfaces.pipeline.IBranchingRule.IBranchingRule

