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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sef.core.artifacts import Frame, Signal
from sef.core.artifacts.buffer import FrameBuffer, SignalBuffer
from sef.core.artifacts.data import TwoDimGraphData
from sef.core.artifacts.signal_sample import BoxSignalSample
from sef.core.interfaces import (
    IAnalyzer,
    IData,
    IFrameBufferProcessor,
    ISignal,
    ISignalExtractor,
    ISignalSample,
    StageCapabilities,
)
from sef.core.interfaces.BufferContracts import IBuffer, IFrameBuffer
from sef.core.interfaces.IFrameExtractor import IFrameExtractor
from sef.core.interfaces.StreamingContracts import (
    IStreamingAnalyzer,
    IStreamingFrameBufferProcessor,
    IStreamingFrameExtractor,
    IStreamingSignalExtractor,
)
from sef.core.pipeline import FluentPipelineBuilder, Pipeline
from sef.core.pipeline.LatencyPolicy import BlockingFrameLatencyPolicy, FrameLatencyPolicy


SCENARIOS = ("direct_loop", "sef_batch", "sef_streaming")
BASELINE_SCENARIO = "direct_loop"
BYTES_PER_RGB_PIXEL = 3
BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Input parameters shared by all framework-overhead scenarios."""

    frame_count: int
    width: int
    height: int
    frame_buffer_size: int
    signal_buffer_size: int
    data_buffer_size: int

    @property
    def frame_bytes(self) -> int:
        """Return the expected byte size of one RGB frame."""
        return self.width * self.height * BYTES_PER_RGB_PIXEL

    @property
    def full_frame_sequence_mib(self) -> float:
        """Return the estimated cost of materializing all source frames."""
        return self.frame_count * self.frame_bytes / BYTES_PER_MIB

    @property
    def frame_queue_mib(self) -> float:
        """Return the estimated bounded frame queue cost for one streaming edge."""
        return self.frame_buffer_size * self.frame_bytes / BYTES_PER_MIB


class BatchSyntheticFrameExtractor(IFrameExtractor):
    """Materialize deterministic frames through the batch extractor contract."""

    capabilities = StageCapabilities.batch(stateful=False, realtime_safe=False)

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__({"benchmark": "framework_overhead_batch_source"})
        self._config = config
        self.resize = (config.width, config.height)
        self.max_frames = config.frame_count

    def extract(self) -> FrameBuffer:
        buffer = FrameBuffer(buffer_size=self._config.frame_count)
        for index in range(self._config.frame_count):
            buffer.put(_synthetic_frame(index, self._config))
        buffer.close()
        return buffer


class StreamingSyntheticFrameExtractor(IStreamingFrameExtractor):
    """Publish deterministic frames through the streaming extractor contract."""

    capabilities = StageCapabilities.streaming(stateful=False, realtime_safe=True)

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__({"benchmark": "framework_overhead_streaming_source"})
        self._config = config
        self.resize = (config.width, config.height)
        self.max_frames = config.frame_count

    def extract(self) -> FrameBuffer:
        buffer = FrameBuffer(buffer_size=self._config.frame_count)
        self.extract_into(buffer, BlockingFrameLatencyPolicy())
        return buffer

    def extract_into(
        self,
        output_buffer: IFrameBuffer,
        latency_policy: FrameLatencyPolicy,
    ) -> None:
        try:
            for index in range(self._config.frame_count):
                latency_policy.publish(_synthetic_frame(index, self._config), output_buffer)
        finally:
            output_buffer.close()


class BatchPixelTransformProcessor(IFrameBufferProcessor):
    """Apply the benchmark transform through the batch processor contract."""

    capabilities = StageCapabilities.batch(stateful=False, realtime_safe=False)

    def process(self, buffer: FrameBuffer) -> FrameBuffer:
        output = buffer.clone_empty()
        for frame in buffer:
            output.put(_transformed_frame(frame))
        output.close()
        return output


class StreamingPixelTransformProcessor(IStreamingFrameBufferProcessor):
    """Apply the benchmark transform while preserving streaming execution."""

    capabilities = StageCapabilities.streaming(stateful=False, realtime_safe=True)

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
                output_buffer.put(_transformed_frame(frame))
        finally:
            output_buffer.close()


class BatchSignalExtractor(ISignalExtractor):
    """Convert processed frames into a materialized signal."""

    capabilities = StageCapabilities.batch(stateful=False, realtime_safe=False)

    def extract(self, buffer: FrameBuffer) -> ISignal:
        return Signal([_signal_sample(frame) for frame in buffer])


class StreamingSignalExtractor(IStreamingSignalExtractor):
    """Convert processed frames into signal samples progressively."""

    capabilities = StageCapabilities.streaming(stateful=False, realtime_safe=True)

    def extract(self, buffer: FrameBuffer) -> ISignal:
        output = SignalBuffer()
        self.extract_into(buffer, output)
        return Signal(list(output.subscribe(0)))

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


class BatchMeanAnalyzer(IAnalyzer):
    """Analyze the complete signal through the batch analyzer contract."""

    capabilities = StageCapabilities.batch(stateful=False, realtime_safe=False)

    def analyze(self, signal: ISignal) -> IData:
        return _mean_graph(signal, mode="sef_batch")


class StreamingMeanAnalyzer(IStreamingAnalyzer):
    """Analyze the signal progressively through the streaming analyzer contract."""

    capabilities = StageCapabilities.streaming(stateful=True, realtime_safe=True)

    def analyze(self, signal: ISignal) -> IData:
        return _mean_graph(signal, mode="sef_streaming")

    def analyze_into(
        self,
        signal: Iterable[ISignalSample],
        output_buffer: IBuffer[IData],
    ) -> IData:
        x_values: list[float] = []
        y_values: list[float] = []
        total = 0.0
        count = 0
        try:
            for sample in signal:
                value = float(sample.metadata["mean_intensity"])
                count += 1
                total += value
                x_values.append(float(sample.frame_index))
                y_values.append(total / count)
                output_buffer.put(
                    TwoDimGraphData(
                        x=[x_values[-1]],
                        y=[y_values[-1]],
                        title="Progressive mean intensity",
                    )
                )
        finally:
            output_buffer.close()
        return TwoDimGraphData(
            x=x_values,
            y=y_values,
            title="Mean intensity",
            metadata={"mode": "sef_streaming", "samples": count},
        )


def build_pipeline(scenario: str, config: BenchmarkConfig) -> Pipeline:
    """Build one SEF benchmark pipeline for the selected runtime style."""
    builder = FluentPipelineBuilder().with_stream_runtime(
        {
            "frame_buffer_size": config.frame_buffer_size,
            "signal_buffer_size": config.signal_buffer_size,
            "data_buffer_size": config.data_buffer_size,
            "latency_policy": {"name": "blocking", "params": {}},
        }
    )

    if scenario == "sef_batch":
        context = (
            builder.with_frame_extractor(BatchSyntheticFrameExtractor(config))
            .add_frame_processor(BatchPixelTransformProcessor())
            .with_signal_extractor(BatchSignalExtractor())
            .add_analyzer(BatchMeanAnalyzer())
            .build_context()
        )
    elif scenario == "sef_streaming":
        context = (
            builder.with_frame_extractor(StreamingSyntheticFrameExtractor(config))
            .add_frame_processor(StreamingPixelTransformProcessor())
            .with_signal_extractor(StreamingSignalExtractor())
            .add_analyzer(StreamingMeanAnalyzer())
            .build_context()
        )
    else:
        raise ValueError(f"Unsupported SEF scenario: {scenario}")

    return Pipeline(context, pipeline_id=f"benchmark-framework-overhead-{scenario}")


def run_direct_workload(config: BenchmarkConfig) -> TwoDimGraphData:
    """
    Execute the same logical workload without SEF runtime orchestration.

    The baseline still uses SEF artifact value objects (`Frame`, `Signal` and
    `BoxSignalSample`) so the comparison focuses on pipeline/runtime overhead
    rather than on replacing the whole data model with raw arrays.
    """
    frames = [_synthetic_frame(index, config) for index in range(config.frame_count)]
    processed_frames = [_transformed_frame(frame) for frame in frames]
    signal = Signal([_signal_sample(frame) for frame in processed_frames])
    return _mean_graph(signal, mode="direct_loop")


def run_child(args: argparse.Namespace) -> None:
    """Run a single scenario once and print one JSON metrics object."""
    config = BenchmarkConfig(
        frame_count=args.frame_count,
        width=args.width,
        height=args.height,
        frame_buffer_size=args.frame_buffer_size,
        signal_buffer_size=args.signal_buffer_size,
        data_buffer_size=args.data_buffer_size,
    )

    plan: dict[str, Any] | None = None
    tracemalloc.start()
    started = time.perf_counter()
    if args.scenario == BASELINE_SCENARIO:
        result = run_direct_workload(config)
        result_count = 1
        sample_count = int(result.metadata["samples"])
    else:
        pipeline = build_pipeline(args.scenario, config)
        plan = pipeline.execution_plan().as_dict()
        outputs = pipeline.run()
        result = outputs.results[0]
        result_count = len(outputs.results)
        sample_count = int(result.metadata["samples"])
    elapsed_seconds = time.perf_counter() - started
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    payload = {
        "scenario": args.scenario,
        "frame_count": config.frame_count,
        "width": config.width,
        "height": config.height,
        "elapsed_seconds": elapsed_seconds,
        "fps": config.frame_count / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "tracemalloc_peak_mib": tracemalloc_peak / BYTES_PER_MIB,
        "process_peak_rss_mib": _process_peak_rss_mib(),
        "result_count": result_count,
        "sample_count": sample_count,
        "uses_sef_pipeline": args.scenario != BASELINE_SCENARIO,
        "uses_streaming_runtime": args.scenario == "sef_streaming",
        "streaming_stage_count": _count_mode(plan, "streaming") if plan else 0,
        "batch_stage_count": _count_mode(plan, "batch") if plan else 0,
        "streamable_end_to_end": bool(plan.get("streamable_end_to_end")) if plan else False,
        "estimated_full_frame_sequence_mib": config.full_frame_sequence_mib,
        "estimated_frame_queue_mib": config.frame_queue_mib,
        "execution_plan": plan or {},
    }
    print(json.dumps(payload, sort_keys=True))


def run_parent(args: argparse.Namespace) -> None:
    """Run all benchmark scenarios, persist raw data, summaries and plots."""
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        total = args.warmup + args.repetitions
        for index in range(total):
            payload = _run_child_process(args, scenario)
            payload["run_index"] = index - args.warmup
            payload["warmup"] = index < args.warmup
            runs.append(payload)
            label = "warmup" if payload["warmup"] else f"run {payload['run_index'] + 1}"
            print(f"{scenario:<14} {label:<8} {payload['elapsed_seconds']:.4f}s {payload['fps']:.2f} fps samples={payload['sample_count']}")

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
    """Create deterministic RGB-like frame data without reading a video file."""
    image = np.empty((config.height, config.width, 3), dtype=np.uint8)
    image[:, :, 0] = index % 256
    image[:, :, 1] = (index * 3) % 256
    image[:, :, 2] = (index * 7) % 256
    return Frame(
        image=image,
        index=index,
        timestamp_seconds=index / 30.0,
        metadata={"source": "synthetic"},
    )


def _transformed_frame(frame: Frame) -> Frame:
    """Perform a deterministic per-frame operation used by all scenarios."""
    image = np.bitwise_xor(frame.image, np.uint8(0b0000_1111))
    return Frame(
        image=image,
        index=frame.index,
        timestamp_seconds=frame.timestamp_seconds,
        metadata=dict(frame.metadata),
    )


def _signal_sample(frame: Frame) -> BoxSignalSample:
    mean_intensity = float(np.mean(frame.image))
    frame_index = int(frame.index or 0)
    return BoxSignalSample(
        frame_index=frame_index,
        box=(0, 0, 1, 1),
        centroid=(float(frame_index), mean_intensity),
        timestamp_seconds=frame.timestamp_seconds,
        metadata={
            **frame.metadata,
            "mean_intensity": mean_intensity,
        },
    )


def _mean_graph(signal: ISignal | Iterable[ISignalSample], *, mode: str) -> TwoDimGraphData:
    x_values: list[float] = []
    y_values: list[float] = []
    total = 0.0
    count = 0
    for sample in signal:
        value = float(sample.metadata["mean_intensity"])
        count += 1
        total += value
        x_values.append(float(sample.frame_index))
        y_values.append(total / count)
    return TwoDimGraphData(
        x=x_values,
        y=y_values,
        title="Mean intensity",
        metadata={"mode": mode, "samples": count},
    )


def _summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = _median([run for run in runs if run["scenario"] == BASELINE_SCENARIO], "elapsed_seconds")
    summaries: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        items = [run for run in runs if run["scenario"] == scenario]
        first = items[0]
        elapsed_median = _median(items, "elapsed_seconds")
        summaries.append(
            {
                "scenario": scenario,
                "repetitions": len(items),
                "frame_count": first["frame_count"],
                "width": first["width"],
                "height": first["height"],
                "elapsed_seconds_median": elapsed_median,
                "elapsed_seconds_min": min(item["elapsed_seconds"] for item in items),
                "elapsed_seconds_max": max(item["elapsed_seconds"] for item in items),
                "fps_median": _median(items, "fps"),
                "overhead_vs_direct_ratio": elapsed_median / baseline if baseline > 0 else 0.0,
                "overhead_vs_direct_seconds": elapsed_median - baseline,
                "tracemalloc_peak_mib_median": _median(items, "tracemalloc_peak_mib"),
                "process_peak_rss_mib_median": _median(items, "process_peak_rss_mib"),
                "sample_count": first["sample_count"],
                "uses_sef_pipeline": first["uses_sef_pipeline"],
                "uses_streaming_runtime": first["uses_streaming_runtime"],
                "streaming_stage_count": first["streaming_stage_count"],
                "batch_stage_count": first["batch_stage_count"],
                "streamable_end_to_end": first["streamable_end_to_end"],
                "estimated_full_frame_sequence_mib": first["estimated_full_frame_sequence_mib"],
                "estimated_frame_queue_mib": first["estimated_frame_queue_mib"],
            }
        )
    return summaries


def _median(items: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(item[key]) for item in items))


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
        output_dir / "fps_median.png",
        labels,
        [item["fps_median"] for item in summary],
        ylabel="Frames per second",
        title="Median throughput",
        plt=plt,
    )
    _bar_plot(
        output_dir / "overhead_vs_direct_ratio.png",
        labels,
        [item["overhead_vs_direct_ratio"] for item in summary],
        ylabel="Ratio",
        title="Median overhead vs direct loop",
        plt=plt,
    )
    _bar_plot(
        output_dir / "process_peak_rss_mib_median.png",
        labels,
        [item["process_peak_rss_mib_median"] for item in summary],
        ylabel="MiB",
        title="Median process peak RSS",
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
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=8)
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
        output_dir / "fps_median.svg",
        labels,
        [item["fps_median"] for item in summary],
        ylabel="Frames per second",
        title="Median throughput",
    )
    _svg_bar_plot(
        output_dir / "overhead_vs_direct_ratio.svg",
        labels,
        [item["overhead_vs_direct_ratio"] for item in summary],
        ylabel="Ratio",
        title="Median overhead vs direct loop",
    )
    _svg_bar_plot(
        output_dir / "process_peak_rss_mib_median.svg",
        labels,
        [item["process_peak_rss_mib_median"] for item in summary],
        ylabel="MiB",
        title="Median process peak RSS",
    )


def _svg_bar_plot(
    path: Path,
    labels: list[str],
    values: list[float],
    *,
    ylabel: str,
    title: str,
) -> None:
    width = 800
    height = 470
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 100
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = ["#0FA4DB", "#0173F1", "#31BDF1"]
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


def _count_mode(plan: dict[str, Any] | None, mode: str) -> int:
    if not plan:
        return 0
    return sum(1 for stage in plan.get("stages", []) if stage.get("execution_mode") == mode)


def _process_peak_rss_mib() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return raw / BYTES_PER_MIB
    return raw / 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark framework overhead by comparing a direct loop with equivalent SEF pipelines.",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scenario", choices=SCENARIOS, help=argparse.SUPPRESS)
    parser.add_argument("--frame-count", type=int, default=1200, help="Synthetic frame count per run.")
    parser.add_argument("--width", type=int, default=160, help="Synthetic frame width.")
    parser.add_argument("--height", type=int, default=90, help="Synthetic frame height.")
    parser.add_argument("--frame-buffer-size", type=int, default=8, help="Streaming frame buffer size.")
    parser.add_argument("--signal-buffer-size", type=int, default=8, help="Streaming signal buffer size.")
    parser.add_argument("--data-buffer-size", type=int, default=8, help="Streaming analyzer data buffer size.")
    parser.add_argument("--repetitions", type=int, default=7, help="Measured repetitions per scenario.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repetitions per scenario.")
    parser.add_argument(
        "--output",
        default="benchmarks/results/framework_overhead",
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.child:
        run_child(args)
    else:
        run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
