from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from library.core.pipeline.PipelineConfigVersioning import CURRENT_PIPELINE_CONFIG_VERSION
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineExportUtils import (
    dotted_path,
    is_rebuildable_param,
    json_dumps,
    to_exportable_data,
    yaml_dumps,
)
from library.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from library.core.pipeline.VisualizerBinding import VisualizerBinding
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry
from library.core.visualization.PipelineOutputs import PipelineOutputs


class PipelineConfigExporter:
    """
    Export an executable PipelineContext as a registry-driven configuration.

    The exporter is deliberately UI-agnostic. It reads only core pipeline
    abstractions, optional source configuration captured by ConfigPipelineBuilder,
    and completed PipelineOutputs for execution/artifact metadata.
    """

    SCHEMA_VERSION = CURRENT_PIPELINE_CONFIG_VERSION
    _PIPELINE_STAGE_KEYS = {
        "frame_extractor",
        "frame_processors",
        "signal_extractor",
        "signal_cleaners",
        "analyzers",
        "visualizers",
        "intermediate_frames",
        "runtime",
    }

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._registry = registry
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def export(
        self,
        context: PipelineContext,
        outputs: PipelineOutputs | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Return a JSON/YAML-safe export containing config and run metadata.

        The top-level ``pipeline`` key remains compatible with
        ConfigPipelineBuilder; additional keys document the executed run.
        """
        pipeline_config = self._pipeline_config(context)
        export: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "exported_at": self._clock().isoformat(),
            "pipeline": pipeline_config,
            "components": self._component_descriptors(context, pipeline_config),
            "execution": self._execution_metadata(outputs, execution_metadata),
            "artifacts": self._artifact_metadata(outputs),
        }
        return to_exportable_data(export)

    def export_json(
        self,
        context: PipelineContext,
        outputs: PipelineOutputs | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Return the pipeline export as formatted JSON."""
        return self.to_json(self.export(context, outputs, execution_metadata))

    def export_yaml(
        self,
        context: PipelineContext,
        outputs: PipelineOutputs | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Return the pipeline export as dependency-free YAML."""
        return self.to_yaml(self.export(context, outputs, execution_metadata))

    @staticmethod
    def to_json(export_config: Mapping[str, Any]) -> str:
        """Serialize a previously built export as formatted JSON."""
        return json_dumps(export_config)

    @staticmethod
    def to_yaml(export_config: Mapping[str, Any]) -> str:
        """Serialize a previously built export as YAML."""
        return yaml_dumps(export_config)

    def _pipeline_config(self, context: PipelineContext) -> dict[str, Any]:
        source_pipeline = self._source_pipeline(context)
        config: dict[str, Any] = {
            "frame_extractor": self._config_entry(
                PluginCategory.FRAME_EXTRACTOR,
                context.frame_extractor,
                source_pipeline.get("frame_extractor"),
            ),
            "frame_processors": self._frame_processor_entries(
                context,
                source_pipeline.get("frame_processors"),
            ),
            "signal_extractor": self._config_entry(
                PluginCategory.SIGNAL_EXTRACTOR,
                context.signal_extractor,
                source_pipeline.get("signal_extractor"),
            ),
            "signal_cleaners": self._component_list_entries(
                PluginCategory.SIGNAL_CLEANER,
                context.signal_cleaners,
                source_pipeline.get("signal_cleaners"),
            ),
            "analyzers": self._component_list_entries(
                PluginCategory.ANALYZER,
                context.analyzers,
                source_pipeline.get("analyzers"),
            ),
            "visualizers": self._visualizer_entries(context, source_pipeline.get("visualizers")),
            "runtime": context.stream_runtime.as_dict(),
        }

        intermediate_frames = self._intermediate_frames_entry(context, source_pipeline.get("intermediate_frames"))
        if intermediate_frames:
            config["intermediate_frames"] = intermediate_frames

        for key, value in source_pipeline.items():
            if key not in self._PIPELINE_STAGE_KEYS:
                config[key] = to_exportable_data(value)

        return config

    def _frame_processor_entries(self, context: PipelineContext, source_entries: Any) -> list[dict[str, Any]]:
        sources = source_entries if isinstance(source_entries, list) else []
        entries: list[dict[str, Any]] = []
        for index, processor in enumerate(context.frame_processors):
            component = self._frame_processor_export_component(processor)
            source = sources[index] if index < len(sources) else None
            if isinstance(processor, SingleFrameProcessorAdapter):
                entry = self._config_entry(PluginCategory.SINGLE_FRAME_PROCESSOR, component, source)
                entry["processor_type"] = "single_frame"
                entry["adapter"] = "SingleFrameProcessorAdapter"
            else:
                entry = self._config_entry(PluginCategory.FRAME_BUFFER_PROCESSOR, component, source)
                entry["processor_type"] = "frame_buffer"
            entries.append(entry)
        return entries

    @staticmethod
    def _frame_processor_export_component(processor: Any) -> Any:
        if isinstance(processor, SingleFrameProcessorAdapter):
            return processor.single_frame_processor
        return processor

    @staticmethod
    def _source_pipeline(context: PipelineContext) -> dict[str, Any]:
        source_config = to_exportable_data(context.source_config)
        if not isinstance(source_config, Mapping):
            return {}
        pipeline = source_config.get("pipeline", {})
        return dict(pipeline) if isinstance(pipeline, Mapping) else {}

    def _component_list_entries(
        self,
        category: PluginCategory,
        components: tuple[Any, ...],
        source_entries: Any,
    ) -> list[dict[str, Any]]:
        sources = source_entries if isinstance(source_entries, list) else []
        return [
            self._config_entry(category, component, sources[index] if index < len(sources) else None)
            for index, component in enumerate(components)
        ]

    def _visualizer_entries(self, context: PipelineContext, source_entries: Any) -> list[dict[str, Any]]:
        sources = source_entries if isinstance(source_entries, list) else []
        unbound_sources = [entry for entry in sources if isinstance(entry, Mapping) and "result_indices" not in entry]
        bound_sources = [entry for entry in sources if isinstance(entry, Mapping) and "result_indices" in entry]

        entries = [
            self._config_entry(
                PluginCategory.VISUALIZER,
                visualizer,
                unbound_sources[index] if index < len(unbound_sources) else None,
            )
            for index, visualizer in enumerate(context.visualizers)
        ]
        entries.extend(
            self._visualizer_binding_entry(
                binding,
                bound_sources[index] if index < len(bound_sources) else None,
            )
            for index, binding in enumerate(context.visualizer_bindings)
        )
        return entries

    def _visualizer_binding_entry(
        self,
        binding: VisualizerBinding,
        source_entry: Any,
    ) -> dict[str, Any]:
        entry = self._config_entry(PluginCategory.VISUALIZER, binding.visualizer, source_entry)
        if binding.result_indices is not None:
            entry["result_indices"] = list(binding.result_indices)
        return entry

    def _intermediate_frames_entry(self, context: PipelineContext, source_entry: Any) -> dict[str, Any]:
        if isinstance(source_entry, Mapping):
            entry = to_exportable_data(source_entry)
            if isinstance(entry, dict):
                return entry

        capture = getattr(context, "intermediate_frame_capture", None)
        visualizers = getattr(context, "intermediate_frame_visualizers", ())
        if capture is None or not getattr(capture, "enabled", False):
            if not visualizers:
                return {}

        entry = capture.to_metadata() if hasattr(capture, "to_metadata") else {}
        if visualizers:
            entry["visualizers"] = [
                self._config_entry(PluginCategory.VISUALIZER, visualizer, None)
                for visualizer in visualizers
            ]
        return entry

    def _config_entry(
        self,
        category: PluginCategory,
        component: Any,
        source_entry: Any,
    ) -> dict[str, Any]:
        if isinstance(source_entry, Mapping) and source_entry.get("name"):
            entry: dict[str, Any] = {"name": str(source_entry["name"])}
            params = source_entry.get("params", {})
            if isinstance(params, Mapping) and params:
                entry["params"] = to_exportable_data(params)
            if "result_indices" in source_entry:
                entry["result_indices"] = to_exportable_data(source_entry["result_indices"])
            return entry

        entry = {"name": self._registered_name(category, component) or type(component).__name__}
        params = self._infer_component_params(component)
        if params:
            entry["params"] = params
        return entry

    def _infer_component_params(self, component: Any) -> dict[str, Any]:
        params: dict[str, Any] = {}
        signature = self._constructor_signature(component)
        config = getattr(component, "config", None)

        for name, parameter in signature.parameters.items():
            if name == "self" or parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                continue
            if name == "config":
                if isinstance(config, Mapping) and config:
                    params["config"] = to_exportable_data(config)
                continue
            if not hasattr(component, name):
                continue

            value = getattr(component, name)
            if not is_rebuildable_param(value):
                continue
            params[name] = to_exportable_data(value)

        if "config" not in params and isinstance(config, Mapping) and config:
            params["config"] = to_exportable_data(config)
        return params

    @staticmethod
    def _constructor_signature(component: Any) -> inspect.Signature:
        try:
            return inspect.signature(type(component).__init__)
        except (TypeError, ValueError):
            return inspect.Signature()

    def _registered_name(self, category: PluginCategory, component: Any) -> str | None:
        if self._registry is None:
            return None

        subclass_match: str | None = None
        for definition in self._registry.list(category):
            factory = definition.factory
            if inspect.isclass(factory):
                if type(component) is factory:
                    return definition.name
                if isinstance(component, factory):
                    subclass_match = subclass_match or definition.name
        return subclass_match

    def _component_descriptors(
        self,
        context: PipelineContext,
        pipeline_config: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        descriptors: list[dict[str, Any]] = []
        order = 0
        order = self._append_descriptor(
            descriptors,
            order,
            stage="frame_extractor",
            stage_index=0,
            category=PluginCategory.FRAME_EXTRACTOR,
            component=context.frame_extractor,
            config_entry=pipeline_config["frame_extractor"],
        )
        frame_processor_entries = list(pipeline_config["frame_processors"])
        for stage_index, processor in enumerate(context.frame_processors):
            config_entry = frame_processor_entries[stage_index]
            category = (
                PluginCategory.FRAME_BUFFER_PROCESSOR
                if config_entry.get("processor_type") == "frame_buffer"
                else PluginCategory.SINGLE_FRAME_PROCESSOR
            )
            order = self._append_descriptor(
                descriptors,
                order,
                stage="frame_processor",
                stage_index=stage_index,
                category=category,
                component=self._frame_processor_export_component(processor),
                config_entry=config_entry,
            )
        order = self._append_descriptor(
            descriptors,
            order,
            stage="signal_extractor",
            stage_index=0,
            category=PluginCategory.SIGNAL_EXTRACTOR,
            component=context.signal_extractor,
            config_entry=pipeline_config["signal_extractor"],
        )
        order = self._append_descriptors(
            descriptors,
            order,
            stage="signal_cleaner",
            category=PluginCategory.SIGNAL_CLEANER,
            components=context.signal_cleaners,
            config_entries=pipeline_config["signal_cleaners"],
        )
        order = self._append_descriptors(
            descriptors,
            order,
            stage="analyzer",
            category=PluginCategory.ANALYZER,
            components=context.analyzers,
            config_entries=pipeline_config["analyzers"],
        )

        visualizer_entries = list(pipeline_config["visualizers"])
        visualizer_components = [*context.visualizers, *(binding.visualizer for binding in context.visualizer_bindings)]
        for stage_index, component in enumerate(visualizer_components):
            order = self._append_descriptor(
                descriptors,
                order,
                stage="visualizer",
                stage_index=stage_index,
                category=PluginCategory.VISUALIZER,
                component=component,
                config_entry=visualizer_entries[stage_index],
            )
        intermediate_frames = pipeline_config.get("intermediate_frames", {})
        intermediate_visualizers = intermediate_frames.get("visualizers", []) if isinstance(intermediate_frames, Mapping) else []
        for stage_index, component in enumerate(getattr(context, "intermediate_frame_visualizers", ())):
            config_entry = (
                intermediate_visualizers[stage_index]
                if stage_index < len(intermediate_visualizers)
                else self._config_entry(PluginCategory.VISUALIZER, component, None)
            )
            order = self._append_descriptor(
                descriptors,
                order,
                stage="intermediate_frame_visualizer",
                stage_index=stage_index,
                category=PluginCategory.VISUALIZER,
                component=component,
                config_entry=config_entry,
            )
        return descriptors

    def _append_descriptors(
        self,
        descriptors: list[dict[str, Any]],
        order: int,
        *,
        stage: str,
        category: PluginCategory,
        components: tuple[Any, ...],
        config_entries: list[dict[str, Any]],
    ) -> int:
        for stage_index, component in enumerate(components):
            order = self._append_descriptor(
                descriptors,
                order,
                stage=stage,
                stage_index=stage_index,
                category=category,
                component=component,
                config_entry=config_entries[stage_index],
            )
        return order

    def _append_descriptor(
        self,
        descriptors: list[dict[str, Any]],
        order: int,
        *,
        stage: str,
        stage_index: int,
        category: PluginCategory,
        component: Any,
        config_entry: Mapping[str, Any],
    ) -> int:
        registered_name = str(config_entry.get("name", type(component).__name__))
        params = config_entry.get("params", {})
        descriptor: dict[str, Any] = {
            "order": order,
            "stage": stage,
            "stage_index": stage_index,
            "category": str(category),
            "registered_name": registered_name,
            "registry_confirmed": self._registry_confirms(category, registered_name, component),
            "class_path": dotted_path(component),
            "params": to_exportable_data(params if isinstance(params, Mapping) else {}),
        }
        if "result_indices" in config_entry:
            descriptor["result_indices"] = to_exportable_data(config_entry["result_indices"])
        descriptors.append(descriptor)
        return order + 1

    def _registry_confirms(self, category: PluginCategory, name: str, component: Any) -> bool:
        if self._registry is None:
            return bool(name and name != type(component).__name__)
        try:
            definition = self._registry.get(category, name)
        except KeyError:
            return False
        factory = definition.factory
        return inspect.isclass(factory) and isinstance(component, factory)

    @staticmethod
    def _execution_metadata(
        outputs: PipelineOutputs | None,
        fallback_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if outputs is None:
            return {
                "pipeline_id": None,
                "generated_at": None,
                "metadata": to_exportable_data(fallback_metadata or {}),
            }

        return {
            "pipeline_id": outputs.metadata.pipeline_id,
            "generated_at": outputs.metadata.generated_at.isoformat(),
            "metadata": to_exportable_data(outputs.metadata.execution_metadata),
            "execution_plan": to_exportable_data(outputs.metadata.execution_plan),
        }

    @staticmethod
    def _artifact_metadata(outputs: PipelineOutputs | None) -> list[dict[str, Any]]:
        if outputs is None:
            return []

        artifacts: list[dict[str, Any]] = []
        ordered_artifacts = [
            *[("final", artifact) for artifact in outputs.final_artifacts],
            *[("debug", artifact) for artifact in outputs.debug_artifacts],
        ]
        for order, (channel, artifact) in enumerate(ordered_artifacts):
            entry: dict[str, Any] = {
                "order": order,
                "channel": channel,
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                "role": str(artifact.role),
                "title": artifact.title,
                "description": artifact.description,
                "class_path": dotted_path(artifact),
                "metadata": to_exportable_data(artifact.metadata),
            }
            PipelineConfigExporter._add_artifact_shape(entry, artifact)
            artifacts.append(to_exportable_data(entry))
        return artifacts

    @staticmethod
    def _add_artifact_shape(entry: dict[str, Any], artifact: Any) -> None:
        mime_type = getattr(artifact, "mime_type", None)
        if mime_type is not None:
            entry["mime_type"] = mime_type

        data = getattr(artifact, "data", None)
        if isinstance(data, bytes | bytearray | memoryview):
            entry["size_bytes"] = len(data)

        path = getattr(artifact, "path", None)
        if path is not None:
            entry["path"] = str(path)
            try:
                entry["size_bytes"] = path.stat().st_size
            except OSError:
                pass

        if hasattr(artifact, "filename_suffix"):
            materialized_path = getattr(artifact, "_materialized_path", None)
            entry["filename_suffix"] = artifact.filename_suffix
            entry["materialized"] = materialized_path is not None
            if materialized_path is not None:
                entry["path"] = str(materialized_path)

        if hasattr(artifact, "columns") and hasattr(artifact, "rows"):
            entry["columns"] = list(artifact.columns)
            entry["row_count"] = len(artifact.rows)

        if hasattr(artifact, "payload"):
            entry["payload_keys"] = list(artifact.payload.keys())

        if hasattr(artifact, "content"):
            entry["content_type"] = getattr(artifact, "content_type", None)
            entry["size_chars"] = len(artifact.content)
