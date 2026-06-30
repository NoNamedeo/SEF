from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.artifacts.data.TwoDimGraphData import TwoDimGraphData
from sef.core.artifacts.Frame import Frame
from sef.core.artifacts.Signal import Signal
from sef.core.artifacts.signal_sample.BoxSignalSample import BoxSignalSample
from sef.core.errors import ConfigSchemaError
from sef.core.interfaces.IAnalyzer import IAnalyzer
from sef.core.interfaces.IFrameExtractor import IFrameExtractor
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalExtractor import ISignalExtractor
from sef.core.pipeline.Pipeline import Pipeline
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineRunOptions import (
    PipelineExecutionPlanLevel,
    PipelineRunOptions,
)


class RunOptionsFrameExtractor(IFrameExtractor):
    """Produce one deterministic frame for run-option tests."""

    def extract(self) -> FrameBuffer:
        buffer = FrameBuffer(buffer_size=1)
        buffer.put(Frame(image=np.zeros((2, 2, 3), dtype=np.uint8), index=0))
        buffer.close()
        return buffer


class RunOptionsSignalExtractor(ISignalExtractor):
    """Convert the test frame into one deterministic signal sample."""

    def extract(self, buffer: FrameBuffer) -> ISignal:
        return Signal(
            [
                BoxSignalSample(
                    frame_index=frame.index or 0,
                    box=(0, 0, 1, 1),
                    centroid=(0.0, 1.0),
                )
                for frame in buffer
            ]
        )


class RunOptionsAnalyzer(IAnalyzer):
    """Return a minimal result without introducing unrelated runtime work."""

    def analyze(self, signal: ISignal) -> TwoDimGraphData:
        return TwoDimGraphData(
            x=[0.0],
            y=[float(len(list(signal)))],
            title="Run options",
        )


def _context() -> PipelineContext:
    return PipelineContext(
        frame_extractor=RunOptionsFrameExtractor(),
        signal_extractor=RunOptionsSignalExtractor(),
        analyzers=[RunOptionsAnalyzer()],
    )


def test_lightweight_run_skips_planner_and_reproducibility_exports() -> None:
    pipeline = Pipeline(_context())

    with patch(
        "sef.core.pipeline.Pipeline.PipelineExecutionPlanner.build",
        side_effect=AssertionError("planner must remain lazy"),
    ):
        outputs = pipeline.run()

    assert outputs.results[0].y == [1.0]
    assert outputs.metadata.execution_plan == {}
    assert outputs.metadata.reproducibility == {}


def test_summary_execution_plan_excludes_per_stage_payloads() -> None:
    outputs = Pipeline(
        _context(),
        run_options=PipelineRunOptions(
            execution_plan=PipelineExecutionPlanLevel.SUMMARY,
        ),
    ).run()

    summary = outputs.metadata.execution_plan
    assert summary["stage_count"] == 3
    assert summary["batch_stage_count"] == 3
    assert summary["streaming_stage_count"] == 0
    assert "stages" not in summary
    assert outputs.metadata.reproducibility == {}


def test_full_execution_plan_can_be_enabled_without_reproducibility() -> None:
    outputs = Pipeline(
        _context(),
        run_options=PipelineRunOptions(
            execution_plan=PipelineExecutionPlanLevel.FULL,
            reproducibility=False,
        ),
    ).run()

    assert len(outputs.metadata.execution_plan["stages"]) == 3
    assert outputs.metadata.reproducibility == {}


def test_reproducibility_can_be_enabled_without_execution_plan() -> None:
    outputs = Pipeline(
        _context(),
        run_options=PipelineRunOptions(reproducibility=True),
    ).run()

    assert outputs.metadata.execution_plan == {}
    assert "config" in outputs.metadata.reproducibility
    assert "json" in outputs.metadata.reproducibility
    assert "yaml" in outputs.metadata.reproducibility
    assert "python_builder_code" in outputs.metadata.reproducibility


def test_run_options_can_be_read_from_config() -> None:
    options = PipelineRunOptions.from_config(
        {
            "run_options": {
                "execution_plan": "summary",
                "reproducibility": True,
            }
        }
    )

    assert options.execution_plan is PipelineExecutionPlanLevel.SUMMARY
    assert options.reproducibility is True


def test_execution_plan_bool_config_maps_to_level() -> None:
    assert (
        PipelineRunOptions.from_config({"run_options": {"execution_plan": True}}).execution_plan
        is PipelineExecutionPlanLevel.FULL
    )
    assert (
        PipelineRunOptions.from_config({"run_options": {"execution_plan": False}}).execution_plan
        is PipelineExecutionPlanLevel.NONE
    )


def test_run_options_merge_cli_requirements_without_downgrading_config() -> None:
    options = PipelineRunOptions(execution_plan=PipelineExecutionPlanLevel.SUMMARY, reproducibility=True)

    merged = options.with_required(execution_plan=PipelineExecutionPlanLevel.FULL, reproducibility=False)

    assert merged.execution_plan is PipelineExecutionPlanLevel.FULL
    assert merged.reproducibility is True


def test_execution_plan_remains_available_on_lightweight_pipeline() -> None:
    pipeline = Pipeline(_context())

    first = pipeline.execution_plan()
    second = pipeline.execution_plan()

    assert first is second
    assert len(first.stages) == 3


def test_run_options_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="execution_plan must be one of"):
        PipelineRunOptions(execution_plan="verbose")

    with pytest.raises(TypeError, match="reproducibility must be a boolean"):
        PipelineRunOptions(reproducibility=1)

    with pytest.raises(ConfigSchemaError, match="run_options.execution_plan"):
        PipelineRunOptions.from_config({"run_options": {"execution_plan": "verbose"}})

    with pytest.raises(ConfigSchemaError, match="Unsupported field 'run_options.diagnostics'"):
        PipelineRunOptions.from_config({"run_options": {"diagnostics": "verbose"}})
