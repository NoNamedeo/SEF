# API: Core Package

`sef.core` is the convenience import surface for the most common stable
contracts: builders, execution facade, selected artifacts, registry contracts,
error types, visualization outputs, and realtime preview handoff values.

::: sef.core
    options:
      members:
        - ConfigPipelineBuilder
        - Pipeline
        - PipelineContext
        - PipelineExecutionPlan
        - ExecutionPlanStage
        - FluentPipelineBuilder
        - StreamRuntimeConfig
        - LatencyPolicyConfig
        - CURRENT_PIPELINE_CONFIG_VERSION
        - PIPELINE_CONFIG_VERSION_KEY
        - PluginRegistry
        - PluginCategory
        - PluginDefinition
        - Frame
        - FrameBuffer
        - PipelineOutputs
        - VisualArtifact
        - TextArtifact
        - VisualizationContext
        - RealtimeFrame
        - IRealtimeFrameSink
        - StageCapabilities
        - SEFError
        - PipelineError
        - PipelineConfigurationError
        - ConfigSchemaError
        - ConfigVersionError
        - PipelineContextError
        - PluginRegistryError
        - InvalidPluginRegistrationError
        - DuplicatePluginRegistrationError
        - PluginResolutionError
        - PluginConstructionError
        - StageErrorContext
        - StageExecutionError
        - PipelineExecutionError
        - StreamRuntimeError
        - StreamAbortedError
      show_if_no_docstring: true

