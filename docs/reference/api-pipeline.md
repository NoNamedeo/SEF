# API: Pipeline Package

The pipeline package exposes construction, planning, execution, runtime policy,
configuration versioning, and asynchronous runner contracts.

## Execution Facades

::: library.core.pipeline.Pipeline.Pipeline

::: library.core.pipeline.PipelineOrchestrator.PipelineOrchestrator

::: library.core.pipeline.ThreadedPipelineRunner.ThreadedPipelineRunner

## Context and Builders

::: library.core.pipeline.PipelineContext.PipelineContext

::: library.core.pipeline.FluentPipelineBuilder.FluentPipelineBuilder

::: library.core.pipeline.ConfigPipelineBuilder.ConfigPipelineBuilder

::: library.core.pipeline.VisualizerBinding.VisualizerBinding

## Planning and Runtime Policy

::: library.core.pipeline.PipelineExecutionPlan.ExecutionPlanStage

::: library.core.pipeline.PipelineExecutionPlan.PipelineExecutionPlan

::: library.core.pipeline.PipelineExecutionPolicy.PipelineExecutionMode

::: library.core.pipeline.PipelineExecutionPolicy.PipelineExecutionDecision

::: library.core.pipeline.PipelineExecutionPolicy.PipelineStagePolicyContext

::: library.core.pipeline.PipelineExecutionPolicy.PipelineExecutionPolicy

::: library.core.pipeline.PipelineExecutionPolicy.DefaultPipelineExecutionPolicy

::: library.core.pipeline.StreamRuntimeConfig.StreamRuntimeConfig

::: library.core.pipeline.LatencyPolicy.LatencyPolicyConfig

::: library.core.pipeline.LatencyPolicy.FrameLatencyPolicy

::: library.core.pipeline.LatencyPolicy.BlockingFrameLatencyPolicy

::: library.core.pipeline.LatencyPolicy.DropNewestFrameLatencyPolicy

::: library.core.pipeline.LatencyPolicy.DropOldestFrameLatencyPolicy

::: library.core.pipeline.LatencyPolicy.AdaptiveSamplingFrameLatencyPolicy

## Configuration Versioning

::: library.core.pipeline.PipelineConfigVersioning.PipelineConfigMigration

::: library.core.pipeline.PipelineConfigVersioning.VersionedPipelineConfig

::: library.core.pipeline.PipelineConfigVersioning.PipelineConfigVersionManager

::: library.core.pipeline.PipelineConfigVersioning.normalize_pipeline_config

## Run Snapshots

::: library.core.pipeline.PipelineRunSnapshot.PipelineRunState

::: library.core.pipeline.PipelineRunSnapshot.PipelineRunSnapshot

