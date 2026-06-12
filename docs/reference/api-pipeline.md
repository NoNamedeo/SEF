# API: Pipeline Package

The pipeline package exposes construction, planning, execution, runtime policy,
configuration versioning, and asynchronous runner contracts.

## Execution Facades

::: sef.core.pipeline.Pipeline.Pipeline

::: sef.core.pipeline.PipelineOrchestrator.PipelineOrchestrator

::: sef.core.pipeline.ThreadedPipelineRunner.ThreadedPipelineRunner

## Context and Builders

::: sef.core.pipeline.PipelineContext.PipelineContext

::: sef.core.pipeline.FluentPipelineBuilder.FluentPipelineBuilder

::: sef.core.pipeline.ConfigPipelineBuilder.ConfigPipelineBuilder

::: sef.core.pipeline.VisualizerBinding.VisualizerBinding

## Planning and Runtime Policy

::: sef.core.pipeline.PipelineExecutionPlan.ExecutionPlanStage

::: sef.core.pipeline.PipelineExecutionPlan.PipelineExecutionPlan

::: sef.core.pipeline.PipelineExecutionPolicy.PipelineExecutionMode

::: sef.core.pipeline.PipelineExecutionPolicy.PipelineExecutionDecision

::: sef.core.pipeline.PipelineExecutionPolicy.PipelineStagePolicyContext

::: sef.core.pipeline.PipelineExecutionPolicy.PipelineExecutionPolicy

::: sef.core.pipeline.PipelineExecutionPolicy.DefaultPipelineExecutionPolicy

::: sef.core.pipeline.StreamRuntimeConfig.StreamRuntimeConfig

::: sef.core.pipeline.LatencyPolicy.LatencyPolicyConfig

::: sef.core.pipeline.LatencyPolicy.FrameLatencyPolicy

::: sef.core.pipeline.LatencyPolicy.BlockingFrameLatencyPolicy

::: sef.core.pipeline.LatencyPolicy.DropNewestFrameLatencyPolicy

::: sef.core.pipeline.LatencyPolicy.DropOldestFrameLatencyPolicy

::: sef.core.pipeline.LatencyPolicy.AdaptiveSamplingFrameLatencyPolicy

## Configuration Versioning

::: sef.core.pipeline.PipelineConfigVersioning.PipelineConfigMigration

::: sef.core.pipeline.PipelineConfigVersioning.VersionedPipelineConfig

::: sef.core.pipeline.PipelineConfigVersioning.PipelineConfigVersionManager

::: sef.core.pipeline.PipelineConfigVersioning.normalize_pipeline_config

## Run Snapshots

::: sef.core.pipeline.PipelineRunSnapshot.PipelineRunState

::: sef.core.pipeline.PipelineRunSnapshot.PipelineRunSnapshot

