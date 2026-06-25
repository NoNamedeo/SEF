from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sef.core.artifacts import Frame, Signal
from sef.core.artifacts.buffer import FrameBuffer
from sef.core.artifacts.data import TwoDimGraphData
from sef.core.artifacts.signal_sample import BoxSignalSample
from sef.core.interfaces import IData, ISignal, ISignalSample, StageCapabilities
from sef.core.interfaces.BufferContracts import IBuffer, IFrameBuffer
from sef.core.interfaces.StreamingContracts import (
    IStreamingAnalyzer,
    IStreamingFrameBufferProcessor,
    IStreamingFrameExtractor,
    IStreamingSignalExtractor,
)
from sef.core.pipeline import FluentPipelineBuilder, Pipeline
from sef.core.pipeline.LatencyPolicy import BlockingFrameLatencyPolicy, FrameLatencyPolicy


BYTES_PER_RGB_PIXEL = 3
BYTES_PER_MIB = 1024 * 1024
PRODUCED_AT_KEY = "produced_at_seconds"


@dataclass(frozen=True, slots=True)
class LatencyScenario:
    """Serializable runtime policy scenario used by the benchmark runner."""

    name: str
    policy_name: str
    policy_params: Mapping[str, Any]


LATENCY_SCENARIOS = (
    LatencyScenario("blocking", "blocking", {}),
    LatencyScenario("drop_newest", "drop_newest", {}),
    LatencyScenario("drop_oldest", "drop_oldest", {}),
    LatencyScenario(
        "adaptive_sampling",
        "adaptive_sampling",
        {
            "min_interval": 1,
            "max_interval": 8,
            "high_watermark": 0.65,
            "low_watermark": 0.20,
        },
    ),
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Input parameters shared by all latency-policy scenarios."""

    frame_count: int
    width: int
    height: int
    frame_buffer_size: int
    signal_buffer_size: int
    data_buffer_size: int
    source_interval_seconds: float
    processor_delay_seconds: float

    @property
    def frame_bytes(self) -> int:
        """Return the expected byte size of one RGB frame."""
        return self.width * self.height * BYTES_PER_RGB_PIXEL

    @property
    def frame_queue_mib(self) -> float:
        """Return the estimated memory cost of one bounded frame queue."""
        return self.frame_buffer_size * self.frame_bytes / BYTES_PER_MIB

    @property
    def offered_source_fps(self) -> float:
        """Return the nominal source rate before runtime backpressure."""
        if self.source_interval_seconds <= 0:
            return 0.0
        return 1.0 / self.source_interval_seconds

    @property
    def nominal_processor_fps(self) -> float:
        """Return the nominal processor rate before scheduler overhead."""
        if self.processor_delay_seconds <= 0:
            return 0.0
        return 1.0 / self.processor_delay_seconds


class RealtimeSyntheticFrameExtractor(IStreamingFrameExtractor):
    """Emit deterministic frames at a configurable source cadence."""

    capabilities = StageCapabilities.streaming(stateful=False, realtime_safe=True)

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__({"benchmark": "latency_policy_source"})
        self._config = config
        self.resize = (config.width, config.height)
        self.max_frames = config.frame_count

    def extract(self) -> FrameBuffer:
        buffer = FrameBuffer(buffer_size=self._config.frame_count + 1)
        for index in range(self._config.frame_count):
            buffer.put(_synthetic_frame(index, self._config))
        buffer.close()
        return buffer

    def extract_into(
        self,
        output_buffer: IFrameBuffer,
        latency_policy: FrameLatencyPolicy,
    ) -> None:
        try:
            for index in range(self._config.frame_count):
                if index > 0 and self._config.source_interval_seconds > 0:
                    time.sleep(self._config.source_interval_seconds)
                latency_policy.publish(_synthetic_frame(index, self._config), output_buffer)
        finally:
            _wait_until_close_can_publish_sentinel(output_buffer)
            output_buffer.close()


class SlowStreamingFrameProcessor(IStreamingFrameBufferProcessor):
    """Simulate a realtime stage that is slower than the incoming frame rate."""

    capabilities = StageCapabilities.streaming(stateful=False, realtime_safe=True)

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config

    def process(self, buffer: FrameBuffer) -> FrameBuffer:
        output = buffer.clone_empty()
        self.process_into(buffer, output, processor_index=0, intermediate_store=None)
        return output

    def process_into(
        self,
        input_buffer: Iterable[Frame],
        output_buffer: IBuffer[Frame],
        *,
        processor_index: int,
        intermediate_store: Any | None,
    ) -> None:
        try:
            for frame in input_buffer:
                if self._config.processor_delay_seconds > 0:
                    time.sleep(self._config.processor_delay_seconds)
                output_buffer.put(_mark_processed(frame))
        finally:
            output_buffer.close()


class StreamingFrameIndexSignalExtractor(IStreamingSignalExtractor):
    """Convert processed frames into signal samples preserving source indexes."""

    capabilities = StageCapabilities.streaming(stateful=False, realtime_safe=True)

    def extract(self, buffer: FrameBuffer) -> ISignal:
        return Signal([_signal_sample(frame) for frame in buffer])

    def extract_into(
        self,
        frames: IFrameBuffer,
        output_buffer: IBuffer[ISignalSample],
    ) -> None:
        try:
            for frame in frames:
                output_buffer.put(_signal_sample(frame))
        finally:
            output_buffer.close()


class FreshnessAnalyzer(IStreamingAnalyzer):
    """Measure downstream coverage, frame freshness and end-to-end latency."""

    capabilities = StageCapabilities.streaming(stateful=True, realtime_safe=True)

    def __init__(self, source_frame_count: int) -> None:
        self._source_frame_count = source_frame_count
        self._last_source_index = max(0, source_frame_count - 1)

    def analyze(self, signal: ISignal) -> IData:
        return self._analyze_samples(signal, output_buffer=None)

    def analyze_into(
        self,
        signal: Iterable[ISignalSample],
        output_buffer: IBuffer[IData],
    ) -> IData:
        try:
            return self._analyze_samples(signal, output_buffer=output_buffer)
        finally:
            output_buffer.close()

    def _analyze_samples(
        self,
        signal: Iterable[ISignalSample],
        *,
        output_buffer: IBuffer[IData] | None,
    ) -> TwoDimGraphData:
        indexes: list[int] = []
        staleness_values: list[float] = []
        latency_values_ms: list[float] = []

        for sample_number, sample in enumerate(signal, start=1):
            frame_index = int(sample.frame_index)
            staleness = float(max(0, self._last_source_index - frame_index))
            latency_ms = _sample_latency_ms(sample)
            indexes.append(frame_index)
            staleness_values.append(staleness)
            if latency_ms is not None:
                latency_values_ms.append(latency_ms)
            if output_buffer is not None:
                output_buffer.put(
                    TwoDimGraphData(
                        x=[float(sample_number)],
                        y=[staleness],
                        title="Progressive frame staleness",
                        x_label="Processed sample",
                        y_label="Staleness (frames)",
                    )
                )

        metadata = _freshness_metrics(
            indexes=indexes,
            staleness_values=staleness_values,
            latency_values_ms=latency_values_ms,
            source_frame_count=self._source_frame_count,
        )
        return TwoDimGraphData(
            x=[float(index + 1) for index in range(len(staleness_values))],
            y=staleness_values,
            title="Frame staleness by processed sample",
            x_label="Processed sample",
            y_label="Staleness (frames)",
            metadata=metadata,
        )


def build_pipeline(scenario: LatencyScenario, config: BenchmarkConfig) -> Pipeline:
    """Build a streaming benchmark pipeline with a policy-specific runtime."""
    runtime_config = {
        "frame_buffer_size": config.frame_buffer_size,
        "signal_buffer_size": config.signal_buffer_size,
        "data_buffer_size": config.data_buffer_size,
        "latency_policy": {
            "name": scenario.policy_name,
            "params": dict(scenario.policy_params),
        },
    }
    context = (
        FluentPipelineBuilder()
        .with_stream_runtime(runtime_config)
        .with_frame_extractor(RealtimeSyntheticFrameExtractor(config))
        .add_frame_processor(SlowStreamingFrameProcessor(config))
        .with_signal_extractor(StreamingFrameIndexSignalExtractor())
        .add_analyzer(FreshnessAnalyzer(config.frame_count))
        .build_context()
    )
    return Pipeline(context, pipeline_id=f"benchmark-latency-{scenario.name}")


def run_child(args: argparse.Namespace) -> None:
    """Run one policy scenario once and print one JSON metrics object."""
    scenario = _scenario_by_name(args.scenario)
    config = BenchmarkConfig(
        frame_count=args.frame_count,
        width=args.width,
        height=args.height,
        frame_buffer_size=args.frame_buffer_size,
        signal_buffer_size=args.signal_buffer_size,
        data_buffer_size=args.data_buffer_size,
        source_interval_seconds=args.source_interval_seconds,
        processor_delay_seconds=args.processor_delay_seconds,
    )
    pipeline = build_pipeline(scenario, config)
    plan = pipeline.execution_plan()

    tracemalloc.start()
    started = time.perf_counter()
    outputs = pipeline.run()
    elapsed_seconds = time.perf_counter() - started
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = outputs.results[0]
    analysis_metrics = dict(getattr(result, "metadata", {}))
    latency_policy_metrics = dict(outputs.metadata.execution_metadata["latency_policy_metrics"])
    processed_frames = int(analysis_metrics["processed_frames"])

    payload = {
        "scenario": scenario.name,
        "latency_policy": scenario.policy_name,
        "latency_params": dict(scenario.policy_params),
        "frame_count": config.frame_count,
        "width": config.width,
        "height": config.height,
        "frame_buffer_size": config.frame_buffer_size,
        "signal_buffer_size": config.signal_buffer_size,
        "data_buffer_size": config.data_buffer_size,
        "source_interval_seconds": config.source_interval_seconds,
        "processor_delay_seconds": config.processor_delay_seconds,
        "offered_source_fps": config.offered_source_fps,
        "nominal_processor_fps": config.nominal_processor_fps,
        "elapsed_seconds": elapsed_seconds,
        "produced_fps": config.frame_count / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "processed_fps": processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "tracemalloc_peak_mib": tracemalloc_peak / BYTES_PER_MIB,
        "process_peak_rss_mib": _process_peak_rss_mib(),
        "accepted_frames": int(latency_policy_metrics.get("accepted_frames", 0)),
        "dropped_frames": int(latency_policy_metrics.get("dropped_frames", 0)),
        "seen_frames": int(latency_policy_metrics.get("seen_frames", config.frame_count)),
        "current_interval": latency_policy_metrics.get("current_interval", ""),
        "policy_drop_ratio": _ratio(
            int(latency_policy_metrics.get("dropped_frames", 0)),
            config.frame_count,
        ),
        "processed_ratio": _ratio(processed_frames, config.frame_count),
        "result_count": len(outputs.results),
        "streaming_stage_count": _count_mode(plan.as_dict(), "streaming"),
        "batch_stage_count": _count_mode(plan.as_dict(), "batch"),
        "streamable_end_to_end": plan.streamable_end_to_end,
        "estimated_frame_queue_mib": config.frame_queue_mib,
        "execution_plan": plan.as_dict(),
        **analysis_metrics,
    }
    print(json.dumps(payload, sort_keys=True))


def run_parent(args: argparse.Namespace) -> None:
    """Run all latency-policy scenarios, persist raw data, summaries and plots."""
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for scenario in LATENCY_SCENARIOS:
        total = args.warmup + args.repetitions
        for index in range(total):
            payload = _run_child_process(args, scenario.name)
            payload["run_index"] = index - args.warmup
            payload["warmup"] = index < args.warmup
            runs.append(payload)
            label = "warmup" if payload["warmup"] else f"run {payload['run_index'] + 1}"
            print(
                f"{scenario.name:<18} {label:<8} "
                f"{payload['elapsed_seconds']:.4f}s "
                f"processed={payload['processed_frames']:>4}/"
                f"{payload['frame_count']} "
                f"dropped={payload['dropped_frames']:>4} "
                f"latency={payload['mean_latency_ms']:.2f}ms "
                f"stale={payload['mean_staleness_frames']:.1f} frames"
            )

    measured_runs = [run for run in runs if not run["warmup"]]
    summary = _summarize(measured_runs)
    _write_csv(output_dir / "runs.csv", runs)
    _write_csv(output_dir / "summary.csv", summary)
    _write_json(output_dir / "summary.json", {"runs": measured_runs, "summary": summary})
    _write_json(output_dir / "execution_plans.json", _execution_plans_by_scenario(runs))
    if not args.no_plots:
        _write_plots(output_dir, summary)

    print()
    print(f"Results written to: {output_dir}")
    print("Use summary.csv for thesis tables and the PNG/SVG files for figures.")


def _run_child_process(args: argparse.Namespace, scenario: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--scenario",
        scenario,
        "--frame-count",
        str(args.frame_count),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--frame-buffer-size",
        str(args.frame_buffer_size),
        "--signal-buffer-size",
        str(args.signal_buffer_size),
        "--data-buffer-size",
        str(args.data_buffer_size),
        "--source-interval-seconds",
        str(args.source_interval_seconds),
        "--processor-delay-seconds",
        str(args.processor_delay_seconds),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _synthetic_frame(index: int, config: BenchmarkConfig) -> Frame:
    """Create a small deterministic RGB frame and capture production time."""
    image = np.empty((config.height, config.width, 3), dtype=np.uint8)
    image[:, :, 0] = index % 256
    image[:, :, 1] = (index * 5) % 256
    image[:, :, 2] = (index * 11) % 256
    return Frame(
        image=image,
        index=index,
        timestamp_seconds=index * config.source_interval_seconds,
        metadata={
            "source": "synthetic_realtime",
            PRODUCED_AT_KEY: time.perf_counter(),
        },
    )


def _wait_until_close_can_publish_sentinel(output_buffer: IFrameBuffer) -> None:
    """
    Avoid attributing a final close-time queue eviction to the latency policy.

    ``FrameBuffer.close()`` guarantees consumer wake-up by inserting a sentinel.
    When the queue is full it may evict one queued frame to make room. This wait
    keeps benchmark drop counts tied to the selected latency policy instead of
    the shutdown mechanics.
    """
    while not output_buffer.closed and output_buffer.fill_ratio() >= 1.0:
        time.sleep(0.0005)


def _mark_processed(frame: Frame) -> Frame:
    metadata = dict(frame.metadata)
    metadata["processor_completed_at_seconds"] = time.perf_counter()
    return Frame(
        image=frame.image,
        index=frame.index,
        timestamp_seconds=frame.timestamp_seconds,
        metadata=metadata,
    )


def _signal_sample(frame: Frame) -> BoxSignalSample:
    frame_index = int(frame.index or 0)
    return BoxSignalSample(
        frame_index=frame_index,
        box=(0, 0, 1, 1),
        centroid=(float(frame_index), float(np.mean(frame.image))),
        timestamp_seconds=frame.timestamp_seconds,
        metadata=dict(frame.metadata),
    )


def _sample_latency_ms(sample: ISignalSample) -> float | None:
    produced_at = sample.metadata.get(PRODUCED_AT_KEY)
    if produced_at is None:
        return None
    return (time.perf_counter() - float(produced_at)) * 1000.0


def _freshness_metrics(
    *,
    indexes: list[int],
    staleness_values: list[float],
    latency_values_ms: list[float],
    source_frame_count: int,
) -> dict[str, Any]:
    if not indexes:
        return {
            "processed_frames": 0,
            "first_frame_index": -1,
            "last_frame_index": -1,
            "index_span": 0,
            "missing_inside_span": 0,
            "max_processed_gap": 0,
            "mean_staleness_frames": float(source_frame_count),
            "last_frame_staleness_frames": float(source_frame_count),
            "freshness_score": 0.0,
            "mean_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "last_latency_ms": 0.0,
        }

    first_index = indexes[0]
    last_index = indexes[-1]
    gaps = [current - previous for previous, current in zip(indexes, indexes[1:], strict=False)]
    last_source_index = max(0, source_frame_count - 1)
    return {
        "processed_frames": len(indexes),
        "first_frame_index": first_index,
        "last_frame_index": last_index,
        "index_span": last_index - first_index + 1,
        "missing_inside_span": sum(max(0, gap - 1) for gap in gaps),
        "max_processed_gap": max(gaps, default=0),
        "mean_staleness_frames": float(statistics.fmean(staleness_values)),
        "last_frame_staleness_frames": float(max(0, last_source_index - last_index)),
        "freshness_score": _ratio(last_index, last_source_index),
        "mean_latency_ms": float(statistics.fmean(latency_values_ms)) if latency_values_ms else 0.0,
        "max_latency_ms": max(latency_values_ms, default=0.0),
        "last_latency_ms": latency_values_ms[-1] if latency_values_ms else 0.0,
    }


def _summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for scenario in LATENCY_SCENARIOS:
        items = [run for run in runs if run["scenario"] == scenario.name]
        first = items[0]
        summaries.append(
            {
                "scenario": scenario.name,
                "latency_policy": first["latency_policy"],
                "latency_params": first["latency_params"],
                "repetitions": len(items),
                "frame_count": first["frame_count"],
                "width": first["width"],
                "height": first["height"],
                "frame_buffer_size": first["frame_buffer_size"],
                "source_interval_seconds": first["source_interval_seconds"],
                "processor_delay_seconds": first["processor_delay_seconds"],
                "offered_source_fps": first["offered_source_fps"],
                "nominal_processor_fps": first["nominal_processor_fps"],
                "elapsed_seconds_median": _median(items, "elapsed_seconds"),
                "elapsed_seconds_min": min(item["elapsed_seconds"] for item in items),
                "elapsed_seconds_max": max(item["elapsed_seconds"] for item in items),
                "produced_fps_median": _median(items, "produced_fps"),
                "processed_fps_median": _median(items, "processed_fps"),
                "accepted_frames_median": _median(items, "accepted_frames"),
                "dropped_frames_median": _median(items, "dropped_frames"),
                "seen_frames_median": _median(items, "seen_frames"),
                "current_interval_median": _median_optional(items, "current_interval"),
                "policy_drop_ratio_median": _median(items, "policy_drop_ratio"),
                "processed_frames_median": _median(items, "processed_frames"),
                "processed_ratio_median": _median(items, "processed_ratio"),
                "mean_latency_ms_median": _median(items, "mean_latency_ms"),
                "max_latency_ms_median": _median(items, "max_latency_ms"),
                "mean_staleness_frames_median": _median(items, "mean_staleness_frames"),
                "last_frame_index_median": _median(items, "last_frame_index"),
                "last_frame_staleness_frames_median": _median(items, "last_frame_staleness_frames"),
                "freshness_score_median": _median(items, "freshness_score"),
                "max_processed_gap_median": _median(items, "max_processed_gap"),
                "tracemalloc_peak_mib_median": _median(items, "tracemalloc_peak_mib"),
                "process_peak_rss_mib_median": _median(items, "process_peak_rss_mib"),
                "streaming_stage_count": first["streaming_stage_count"],
                "batch_stage_count": first["batch_stage_count"],
                "streamable_end_to_end": first["streamable_end_to_end"],
                "estimated_frame_queue_mib": first["estimated_frame_queue_mib"],
            }
        )
    return summaries


def _scenario_by_name(name: str) -> LatencyScenario:
    for scenario in LATENCY_SCENARIOS:
        if scenario.name == name:
            return scenario
    raise ValueError(f"Unsupported scenario: {name}")


def _median(items: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(item[key]) for item in items))


def _median_optional(items: list[dict[str, Any]], key: str) -> float | str:
    values = [item[key] for item in items if item.get(key) != ""]
    if not values:
        return ""
    return float(statistics.median(float(value) for value in values))


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _execution_plans_by_scenario(runs: list[dict[str, Any]]) -> dict[str, Any]:
    plans: dict[str, Any] = {}
    for run in runs:
        plans.setdefault(run["scenario"], run["execution_plan"])
    return plans


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    serializable_rows = [{key: _csv_value(value) for key, value in row.items() if key != "execution_plan"} for row in rows]
    fieldnames = list(serializable_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable_rows)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_plots(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_svg_plots(output_dir, summary)
        print("Matplotlib not installed: wrote SVG fallback plots.")
        return

    labels = [item["scenario"] for item in summary]
    _bar_plot(
        output_dir / "elapsed_seconds_median.png",
        labels,
        [item["elapsed_seconds_median"] for item in summary],
        ylabel="Seconds",
        title="Median execution time",
        plt=plt,
    )
    _bar_plot(
        output_dir / "processed_ratio_median.png",
        labels,
        [item["processed_ratio_median"] for item in summary],
        ylabel="Processed / produced",
        title="Median downstream coverage",
        plt=plt,
    )
    _bar_plot(
        output_dir / "policy_drop_ratio_median.png",
        labels,
        [item["policy_drop_ratio_median"] for item in summary],
        ylabel="Dropped / produced",
        title="Median policy drop ratio",
        plt=plt,
    )
    _bar_plot(
        output_dir / "mean_latency_ms_median.png",
        labels,
        [item["mean_latency_ms_median"] for item in summary],
        ylabel="Milliseconds",
        title="Median mean latency",
        plt=plt,
    )
    _bar_plot(
        output_dir / "last_frame_staleness_frames_median.png",
        labels,
        [item["last_frame_staleness_frames_median"] for item in summary],
        ylabel="Frames",
        title="Median final-frame staleness",
        plt=plt,
    )


def _bar_plot(
    path: Path,
    labels: list[str],
    values: list[float],
    *,
    ylabel: str,
    title: str,
    plt: Any,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=12)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_svg_plots(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    labels = [item["scenario"] for item in summary]
    _svg_bar_plot(
        output_dir / "elapsed_seconds_median.svg",
        labels,
        [item["elapsed_seconds_median"] for item in summary],
        ylabel="Seconds",
        title="Median execution time",
    )
    _svg_bar_plot(
        output_dir / "processed_ratio_median.svg",
        labels,
        [item["processed_ratio_median"] for item in summary],
        ylabel="Processed / produced",
        title="Median downstream coverage",
    )
    _svg_bar_plot(
        output_dir / "policy_drop_ratio_median.svg",
        labels,
        [item["policy_drop_ratio_median"] for item in summary],
        ylabel="Dropped / produced",
        title="Median policy drop ratio",
    )
    _svg_bar_plot(
        output_dir / "mean_latency_ms_median.svg",
        labels,
        [item["mean_latency_ms_median"] for item in summary],
        ylabel="Milliseconds",
        title="Median mean latency",
    )
    _svg_bar_plot(
        output_dir / "last_frame_staleness_frames_median.svg",
        labels,
        [item["last_frame_staleness_frames_median"] for item in summary],
        ylabel="Frames",
        title="Median final-frame staleness",
    )


def _svg_bar_plot(
    path: Path,
    labels: list[str],
    values: list[float],
    *,
    ylabel: str,
    title: str,
) -> None:
    width = 820
    height = 470
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 105
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = ["#0389EA", "#0173F1", "#31BDF1", "#0FA4DB"]
    max_value = max(max(values), 1.0)
    bar_width = plot_width / max(len(values), 1) * 0.56
    slot_width = plot_width / max(len(values), 1)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{_xml(title)}</text>',
        f'<text x="28" y="{height / 2:.1f}" transform="rotate(-90 28 {height / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="14">{_xml(ylabel)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>',
    ]
    for tick in range(5):
        ratio = tick / 4
        y = margin_top + plot_height - ratio * plot_height
        value = ratio * max_value
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        lines.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#333">{value:.2f}</text>'
        )

    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        bar_height = (value / max_value) * plot_height
        x = margin_left + index * slot_width + (slot_width - bar_width) / 2
        y = margin_top + plot_height - bar_height
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colors[index % len(colors)]}"/>')
        lines.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{max(y - 8, margin_top + 14):.1f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#111">{value:.2f}</text>'
        )
        lines.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{margin_top + plot_height + 28:.1f}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111">{_xml(label)}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _xml(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _count_mode(plan: dict[str, Any], mode: str) -> int:
    return sum(1 for stage in plan.get("stages", []) if stage.get("execution_mode") == mode)


def _process_peak_rss_mib() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return raw / BYTES_PER_MIB
    return raw / 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SEF realtime latency policies under synthetic backpressure.",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--scenario",
        choices=[scenario.name for scenario in LATENCY_SCENARIOS],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--frame-count", type=int, default=240, help="Synthetic frame count per run.")
    parser.add_argument("--width", type=int, default=160, help="Synthetic frame width.")
    parser.add_argument("--height", type=int, default=90, help="Synthetic frame height.")
    parser.add_argument("--frame-buffer-size", type=int, default=8, help="Streaming frame buffer size.")
    parser.add_argument("--signal-buffer-size", type=int, default=8, help="Streaming signal buffer size.")
    parser.add_argument("--data-buffer-size", type=int, default=8, help="Streaming analyzer data buffer size.")
    parser.add_argument(
        "--source-interval-seconds",
        type=float,
        default=0.0005,
        help="Delay between synthetic source frames before backpressure.",
    )
    parser.add_argument(
        "--processor-delay-seconds",
        type=float,
        default=0.003,
        help="Per-frame delay in the synthetic slow processor.",
    )
    parser.add_argument("--repetitions", type=int, default=5, help="Measured repetitions per policy.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repetitions per policy.")
    parser.add_argument(
        "--output",
        default="benchmarks/results/latency_policy",
        help="Directory where CSV, JSON and PNG/SVG files are written.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG/SVG chart generation.")
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.child and not args.scenario:
        raise SystemExit("--child requires --scenario")
    for field_name in (
        "frame_count",
        "width",
        "height",
        "frame_buffer_size",
        "signal_buffer_size",
        "data_buffer_size",
        "repetitions",
    ):
        if int(getattr(args, field_name)) <= 0:
            raise SystemExit(f"--{field_name.replace('_', '-')} must be greater than zero")
    if int(args.warmup) < 0:
        raise SystemExit("--warmup cannot be negative")
    if float(args.source_interval_seconds) < 0:
        raise SystemExit("--source-interval-seconds cannot be negative")
    if float(args.processor_delay_seconds) < 0:
        raise SystemExit("--processor-delay-seconds cannot be negative")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.child:
        run_child(args)
    else:
        run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
