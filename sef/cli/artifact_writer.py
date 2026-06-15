from __future__ import annotations

import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sef.cli.diagnostics import DiagnosticItem
from sef.core.pipeline.PipelineExecutionPlan import PipelineExecutionPlan
from sef.core.pipeline.PipelineExportUtils import json_dumps, to_exportable_data
from sef.core.visualization.PipelineOutputs import PipelineOutputs
from sef.core.visualization.VisualArtifact import (
    DeferredVideoArtifact,
    ImageArtifact,
    JsonArtifact,
    TableArtifact,
    TextArtifact,
    VideoArtifact,
    VideoFileArtifact,
    VisualArtifact,
)


class ArtifactWriter:
    """Persists run summaries, normalized config, execution plans, and artifacts."""

    def __init__(self, output_dir: Path | str) -> None:
        self._output_dir = Path(output_dir)
        self._artifact_dir = self._output_dir / "artifacts"
        self.warnings: list[DiagnosticItem] = []

    def write(
        self,
        *,
        outputs: PipelineOutputs | None,
        config: Mapping[str, Any],
        execution_plan: PipelineExecutionPlan,
        dry_run: bool = False,
    ) -> list[Path]:
        """Persist supported output files and return created paths."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []
        created.append(self._write_json("config.normalized.json", dict(config)))
        created.append(self._write_json("execution_plan.json", execution_plan.as_dict()))
        created.append(self._write_text("execution_plan.txt", execution_plan.as_text()))

        if outputs is not None:
            created.append(self._write_json("summary.json", self._summary(outputs, dry_run=dry_run)))
            created.extend(self._write_artifacts(outputs))
            reproducibility = outputs.metadata.reproducibility
            if reproducibility.get("yaml"):
                created.append(self._write_text("reproducibility.pipeline.yaml", str(reproducibility["yaml"])))
            if reproducibility.get("python_builder_code"):
                created.append(self._write_text("reproducibility.pipeline.py", str(reproducibility["python_builder_code"])))
        else:
            created.append(
                self._write_json(
                    "summary.json",
                    {
                        "dry_run": True,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "results": 0,
                        "artifacts": 0,
                    },
                )
            )
        return created

    def _write_artifacts(self, outputs: PipelineOutputs) -> list[Path]:
        created: list[Path] = []
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        for channel, artifacts in (("final", outputs.final_artifacts), ("debug", outputs.debug_artifacts)):
            for index, artifact in enumerate(artifacts):
                path = self._write_artifact(channel, index, artifact)
                if path is not None:
                    created.append(path)
        return created

    def _write_artifact(self, channel: str, index: int, artifact: VisualArtifact) -> Path | None:
        stem = self._artifact_stem(channel, index, artifact)
        try:
            if isinstance(artifact, TextArtifact):
                suffix = ".txt" if artifact.content_type == "text/plain" else ".md"
                return self._write_text(self._artifact_name(stem, suffix), artifact.content, base_dir=self._artifact_dir)
            if isinstance(artifact, JsonArtifact):
                return self._write_json(self._artifact_name(stem, ".json"), dict(artifact.payload), base_dir=self._artifact_dir)
            if isinstance(artifact, TableArtifact):
                payload = {"columns": list(artifact.columns), "rows": [dict(row) for row in artifact.rows]}
                return self._write_json(self._artifact_name(stem, ".table.json"), payload, base_dir=self._artifact_dir)
            if isinstance(artifact, ImageArtifact):
                return self._write_bytes(self._artifact_name(stem, self._mime_suffix(artifact.mime_type, ".png")), artifact.data)
            if isinstance(artifact, VideoArtifact):
                return self._write_bytes(self._artifact_name(stem, self._mime_suffix(artifact.mime_type, ".mp4")), artifact.data)
            if isinstance(artifact, VideoFileArtifact):
                suffix = Path(artifact.path).suffix or self._mime_suffix(artifact.mime_type, ".mp4")
                target = self._artifact_dir / self._artifact_name(stem, suffix)
                shutil.copy2(artifact.path, target)
                return target
            if isinstance(artifact, DeferredVideoArtifact):
                materialized = artifact.materialize(self._artifact_dir)
                target = self._artifact_dir / self._artifact_name(stem, Path(materialized).suffix or ".mp4")
                if Path(materialized) != target:
                    shutil.copy2(materialized, target)
                return target
        except Exception as exc:  # noqa: BLE001 - artifact export should be best-effort.
            self.warnings.append(
                DiagnosticItem(
                    "warning",
                    f"Could not export artifact {artifact.artifact_id}.",
                    cause=str(exc),
                    suggestion="Inspect the artifact manually or re-run with --debug for the pipeline traceback.",
                )
            )
            return None

        self.warnings.append(
            DiagnosticItem(
                "warning",
                f"Unsupported artifact type {type(artifact).__name__} for {artifact.artifact_id}.",
                suggestion="The run completed; only this artifact was skipped by the CLI writer.",
            )
        )
        return None

    def _write_json(self, filename: str, payload: Mapping[str, Any], *, base_dir: Path | None = None) -> Path:
        target = (base_dir or self._output_dir) / filename
        target.write_text(json_dumps(payload), encoding="utf-8")
        return target

    def _write_text(self, filename: str, content: str, *, base_dir: Path | None = None) -> Path:
        target = (base_dir or self._output_dir) / filename
        target.write_text(content if content.endswith("\n") else f"{content}\n", encoding="utf-8")
        return target

    def _write_bytes(self, filename: str, data: bytes) -> Path:
        target = self._artifact_dir / filename
        target.write_bytes(data)
        return target

    @staticmethod
    def _summary(outputs: PipelineOutputs, *, dry_run: bool) -> dict[str, Any]:
        return {
            "dry_run": dry_run,
            "pipeline_id": outputs.metadata.pipeline_id,
            "generated_at": outputs.metadata.generated_at.isoformat(),
            "results": len(outputs.results),
            "artifacts": outputs.artifact_count,
            "final_artifacts": len(outputs.final_artifacts),
            "debug_artifacts": len(outputs.debug_artifacts),
            "execution_metadata": to_exportable_data(outputs.metadata.execution_metadata),
        }

    @staticmethod
    def _artifact_stem(channel: str, index: int, artifact: VisualArtifact) -> str:
        safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in artifact.artifact_id)
        return f"{channel}_{index:02d}_{artifact.kind}_{safe_id}"

    @staticmethod
    def _artifact_name(stem: str, suffix: str) -> str:
        return f"{stem}{suffix}"

    @staticmethod
    def _mime_suffix(mime_type: str, default: str) -> str:
        mapping = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "application/json": ".json",
        }
        return mapping.get(mime_type.lower(), default)
