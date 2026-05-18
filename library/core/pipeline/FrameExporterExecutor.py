from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.IFrameExporter import FrameExportContext
from library.core.interfaces.StreamingContracts import IStreamingFrameExporter
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineExecutionResult import ThreadedStageTask
from library.core.pipeline.PipelineStageExecutor import PipelineStageExecutor
from library.core.visualization.VisualArtifact import VisualArtifact


class FrameExporterExecutor:
    """
    Runs frame exporters in batch or streaming mode.

    Exporters are kept separate from frame processors because they produce
    user-facing artifacts while also forwarding frames to the signal tail.
    """

    def __init__(
        self,
        *,
        context: PipelineContext,
        stage_executor: PipelineStageExecutor,
        pipeline_id: str | None,
        execution_metadata: Mapping[str, Any],
    ) -> None:
        self._context = context
        self._stage_executor = stage_executor
        self._pipeline_id = pipeline_id
        self._execution_metadata = dict(execution_metadata)

    def run_batch(self, buffer: FrameBuffer) -> tuple[FrameBuffer, list[VisualArtifact]]:
        """Run exporters after frame processing has been materialized."""
        artifacts: list[VisualArtifact] = []
        current_buffer = buffer
        for exporter_index, exporter in enumerate(self._context.frame_exporters):
            result = self._stage_executor.run(
                f"frame_export[{exporter_index}]",
                lambda e=exporter, b=current_buffer: e.export(b, self._context_for(e)),
            )
            current_buffer = result.buffer
            artifacts.extend(result.artifacts)
        return current_buffer, artifacts

    def build_streaming_tasks(
        self,
        source_buffer: FrameBuffer,
        frame_buffers: list[FrameBuffer],
        artifacts: list[VisualArtifact],
        artifact_lock: Any,
    ) -> tuple[FrameBuffer, list[ThreadedStageTask]]:
        """Create exporter tasks that write artifacts while forwarding frames."""
        current_buffer = source_buffer
        tasks: list[ThreadedStageTask] = []
        for exporter_index, exporter in enumerate(self._context.frame_exporters):
            if not isinstance(exporter, IStreamingFrameExporter):
                raise TypeError(f"{type(exporter).__name__} does not implement IStreamingFrameExporter.")
            output_buffer = FrameBuffer(buffer_size=self._context.stream_runtime.frame_buffer_size)
            frame_buffers.append(output_buffer)
            tasks.append(
                self._streaming_exporter_task(
                    current_buffer,
                    output_buffer,
                    exporter=exporter,
                    exporter_index=exporter_index,
                    artifacts=artifacts,
                    artifact_lock=artifact_lock,
                )
            )
            current_buffer = output_buffer
        return current_buffer, tasks

    def _streaming_exporter_task(
        self,
        input_buffer: FrameBuffer,
        output_buffer: FrameBuffer,
        *,
        exporter: IStreamingFrameExporter,
        exporter_index: int,
        artifacts: list[VisualArtifact],
        artifact_lock: Any,
    ) -> ThreadedStageTask:
        def submit_exporter(executor: ThreadPoolExecutor) -> Future:
            return executor.submit(
                lambda: self._extend_artifacts(
                    artifacts,
                    artifact_lock,
                    self._stage_executor.run(
                        f"frame_export[{exporter_index}]",
                        lambda: exporter.export_into(input_buffer, output_buffer, self._context_for(exporter)),
                    ),
                )
            )

        return submit_exporter

    def _context_for(self, exporter: Any) -> FrameExportContext:
        return FrameExportContext(
            pipeline_id=self._pipeline_id,
            exporter_name=type(exporter).__name__,
            execution_metadata=dict(self._execution_metadata),
        )

    @staticmethod
    def _extend_artifacts(
        artifacts: list[VisualArtifact],
        artifact_lock: Any,
        rendered: tuple[VisualArtifact, ...],
    ) -> None:
        with artifact_lock:
            artifacts.extend(rendered)
