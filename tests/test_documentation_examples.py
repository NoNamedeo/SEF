from __future__ import annotations

import subprocess
import sys

from examples.minimal_pipeline import run_example
from sef.core import ConfigPipelineBuilder, Pipeline, PluginRegistry
from sef.core.artifacts.data import TwoDimGraphData
from sef.core.interfaces import IAnalyzer, ISignal
from sef.core.pipeline import CURRENT_PIPELINE_CONFIG_VERSION
from sef.core.plugins import PluginCategory
from sef.core.visualization import TextArtifact


def test_documented_public_imports_resolve_to_contract_objects() -> None:
    assert ConfigPipelineBuilder is not None
    assert Pipeline is not None
    assert PluginRegistry is not None
    assert PluginCategory.ANALYZER == "analyzer"
    assert CURRENT_PIPELINE_CONFIG_VERSION == "1.0"
    assert issubclass(IAnalyzer, object)
    assert issubclass(ISignal, object)
    assert TwoDimGraphData is not None
    assert TextArtifact is not None


def test_minimal_pipeline_example_runs_as_imported_function() -> None:
    outputs = run_example(frame_count=3)

    assert len(outputs.results) == 1
    assert outputs.artifact_count == 1
    assert outputs.results[0].y == [3.0]
    assert outputs.final_artifacts[0].content == "Sample count: 3.0"


def test_minimal_pipeline_example_runs_as_module() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "examples.minimal_pipeline"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "results: 1" in completed.stdout
    assert "artifacts: 1" in completed.stdout
    assert "sample_count: 3.0" in completed.stdout
    assert "summary: Sample count: 3.0" in completed.stdout
