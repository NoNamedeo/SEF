"""Adapters that convert pipeline outputs into explicit UI view models."""

from __future__ import annotations

import hashlib
from typing import Any

from library.core.visualization.PipelineOutputs import PipelineOutputs
from library.core.visualization.VisualArtifact import VIDEO_ARTIFACT_TYPES
from ui.models.pipeline_outputs import (
    AnalysisResultOutput,
    ArtifactOutput,
    ExecutionResultsView,
    ReconstructedVideoOutput,
)
from ui.state import session


def build_execution_results_view(outputs: PipelineOutputs) -> ExecutionResultsView:
    """Adapt raw pipeline outputs to explicit UI sections."""
    warnings: list[str] = []
    visualizer_outputs: list[ArtifactOutput] = []
    reconstructed_videos: list[ReconstructedVideoOutput] = []

    for artifact in outputs.artifacts:
        if isinstance(artifact, VIDEO_ARTIFACT_TYPES):
            reconstructed_videos.append(
                ReconstructedVideoOutput(
                    artifact_id=artifact.artifact_id,
                    title=artifact.title or "Reconstructed video",
                    artifact=artifact,
                    source="pipeline artifact",
                    metadata=dict(artifact.metadata),
                )
            )
        else:
            visualizer_outputs.append(ArtifactOutput(artifact=artifact, source="pipeline artifact"))

    analysis_results = [
        _build_analysis_result_output(result, idx, warnings)
        for idx, result in enumerate(outputs.results)
    ]

    if not reconstructed_videos:
        for idx, result in enumerate(outputs.results):
            videos, extra_artifacts, extra_warnings = _build_tracking_video_outputs(result, idx)
            reconstructed_videos.extend(videos)
            visualizer_outputs.extend(extra_artifacts)
            warnings.extend(extra_warnings)

    return ExecutionResultsView(
        analysis_results=tuple(analysis_results),
        visualizer_outputs=tuple(visualizer_outputs),
        reconstructed_videos=tuple(reconstructed_videos),
        metadata={
            "pipeline_id": outputs.metadata.pipeline_id,
            "generated_at": outputs.metadata.generated_at.isoformat(),
            "execution_metadata": dict(outputs.metadata.execution_metadata),
        },
        warnings=tuple(warnings),
    )


def _build_analysis_result_output(result: Any, idx: int, warnings: list[str]) -> AnalysisResultOutput:
    """Build the UI representation for a single analysis result."""
    title = getattr(result, "title", None) or f"Risultato {idx + 1}"
    metadata = dict(getattr(result, "metadata", {}) or {})
    type_name = type(result).__name__

    try:
        from library.core.artifacts.CategoryData import CategoryData
        from library.core.artifacts.TrackingPlaybackData import TrackingPlaybackData
        from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
    except Exception:
        CategoryData = None
        TrackingPlaybackData = None
        TwoDimGraphData = None

    if TwoDimGraphData is not None and isinstance(result, TwoDimGraphData):
        preview_artifacts = _safe_render_preview(
            "matplotlib_function",
            lambda: _render_function_preview(result),
            warnings,
        )
        summary = {
            "Campioni": len(result.x),
            f"Min {result.y_label}": f"{min(result.y):.2f}" if result.y else "-",
            f"Max {result.y_label}": f"{max(result.y):.2f}" if result.y else "-",
            f"Media {result.y_label}": f"{(sum(result.y) / len(result.y)):.2f}" if result.y else "-",
        }
        return AnalysisResultOutput(
            result_id=f"result_{idx}_{type_name}",
            title=title,
            type_name=type_name,
            data=result,
            preview_artifacts=preview_artifacts,
            summary=summary,
            metadata=metadata,
        )

    if CategoryData is not None and isinstance(result, CategoryData):
        preview_artifacts = _safe_render_preview(
            "matplotlib_histogram",
            lambda: _render_category_preview(result),
            warnings,
        )
        detail_rows = tuple(
            {
                "track_id": track_id,
                "barriere_attraversate": ", ".join(categories),
            }
            for track_id, categories in sorted(result.track_categories.items())
        )
        summary = {category: result.category_counts.get(category, 0) for category in result.categories}
        return AnalysisResultOutput(
            result_id=f"result_{idx}_{type_name}",
            title=title,
            type_name=type_name,
            data=result,
            preview_artifacts=preview_artifacts,
            summary=summary,
            metadata=metadata,
            detail_rows=detail_rows,
        )

    trajectory_preview = _maybe_render_trajectory_preview(result, warnings)
    if trajectory_preview is not None:
        return AnalysisResultOutput(
            result_id=f"result_{idx}_{type_name}",
            title=title,
            type_name=type_name,
            data=result,
            preview_artifacts=trajectory_preview,
            metadata=metadata,
        )

    if TrackingPlaybackData is not None and isinstance(result, TrackingPlaybackData):
        track_ids = sorted({track.track_id for frame in result.frames for track in frame.tracks})
        summary = {
            "Frame campionati": len(result.frames),
            "Track unici": len(track_ids),
            "Resize": f"{result.resize[0]}x{result.resize[1]}" if result.resize else "originale",
        }
        return AnalysisResultOutput(
            result_id=f"result_{idx}_{type_name}",
            title=title,
            type_name=type_name,
            data=result,
            summary=summary,
            metadata=metadata,
        )

    return AnalysisResultOutput(
        result_id=f"result_{idx}_{type_name}",
        title=title,
        type_name=type_name,
        data=result,
        metadata=metadata,
    )


def _build_tracking_video_outputs(
    result: Any,
    idx: int,
) -> tuple[tuple[ReconstructedVideoOutput, ...], tuple[ArtifactOutput, ...], tuple[str, ...]]:
    """Generate explicit reconstructed videos from tracking playback data if needed."""
    try:
        from library.core.artifacts.TrackingPlaybackData import TrackingPlaybackData
    except Exception:
        return (), (), ()

    if not isinstance(result, TrackingPlaybackData):
        return (), (), ()

    try:
        from library.visualizers.TrackingVideoVisualizer import TrackingVideoVisualizer
    except Exception as exc:
        return (), (), (f"TrackingVideoVisualizer non disponibile: {exc}",)

    cache = dict(session.get(session.TRACKING_VIDEO_CACHE, {}))
    cache_key = _tracking_video_cache_key(result, idx)
    cached_entry = cache.get(cache_key)
    if cached_entry is None:
        try:
            rendered_artifacts = tuple(TrackingVideoVisualizer().render(result))
        except Exception as exc:
            return (), (), (f"Video tracking non disponibile per `{result.title or idx}`: {exc}",)
        cache[cache_key] = {"artifacts": rendered_artifacts}
        session.put(session.TRACKING_VIDEO_CACHE, cache)
    else:
        rendered_artifacts = tuple(cached_entry.get("artifacts", ()))

    videos: list[ReconstructedVideoOutput] = []
    extra_artifacts: list[ArtifactOutput] = []
    warnings: list[str] = []

    for artifact in rendered_artifacts:
        if isinstance(artifact, VIDEO_ARTIFACT_TYPES):
            videos.append(
                ReconstructedVideoOutput(
                    artifact_id=artifact.artifact_id,
                    title=artifact.title or f"Tracking playback {idx + 1}",
                    artifact=artifact,
                    source="tracking_playback preview",
                    metadata=dict(artifact.metadata),
                )
            )
        else:
            extra_artifacts.append(ArtifactOutput(artifact=artifact, source="tracking_playback preview"))

    if not videos:
        warnings.append(f"Nessun video ricostruito generato per `{result.title or idx}`.")

    return tuple(videos), tuple(extra_artifacts), tuple(warnings)


def _safe_render_preview(
    preview_name: str,
    renderer,
    warnings: list[str],
) -> tuple:
    """Render a UI preview without failing the whole result view."""
    try:
        return tuple(renderer())
    except Exception as exc:
        warnings.append(f"Anteprima `{preview_name}` non disponibile: {exc}")
        return ()


def _render_function_preview(result: Any):
    from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer

    return MatplotlibFunctionVisualizer(config={"show": False, "show_scatter": True}).render(result)


def _render_category_preview(result: Any):
    from library.visualizers.MatplotlibHistogramVisualizer import MatplotlibHistogramVisualizer

    return MatplotlibHistogramVisualizer(config={"show": False}).render(result)


def _maybe_render_trajectory_preview(result: Any, warnings: list[str]):
    try:
        from library.core.artifacts.TrajectoryData import TrajectoryData
        from library.visualizers.MatplotlibTrajectoryVisualizer import MatplotlibTrajectoryVisualizer
    except Exception:
        return None

    if not isinstance(result, TrajectoryData):
        return None

    return _safe_render_preview(
        "matplotlib_trajectory",
        lambda: MatplotlibTrajectoryVisualizer(config={"show": False}).render(result),
        warnings,
    )


def _tracking_video_cache_key(result: Any, idx: int) -> str:
    frame_indexes = [frame.frame_index for frame in result.frames]
    digest = hashlib.sha1(
        (
            f"{idx}|{result.source_path}|{result.resize}|{result.fps}|"
            f"{len(frame_indexes)}|{frame_indexes[:5]}|{frame_indexes[-5:] if frame_indexes else []}"
        ).encode("utf-8")
    ).hexdigest()
    return digest
