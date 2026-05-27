from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING
from typing import Any

import cv2
import numpy as np

from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.MaskArtifacts import IntermediateFrameArtifact
from library.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from library.core.interfaces.StageCapabilities import StageCapabilities

if TYPE_CHECKING:
    from library.core.pipeline.FrameProcessingStage import FrameProcessorExecutionContext

_METADATA_KEY = "phase_magnification"
_DEFAULT_CODEC = "MJPG"
_DEFAULT_INPUT_NAME = "input.avi"
_DEFAULT_OUTPUT_NAME = "magnified.avi"

log = logging.getLogger(__name__)


class PhaseMagnificationFrameProcessor(IFrameBufferProcessor):
    """
    Wrap the external MATLAB/Octave phase-magnification reference implementation.

    This processor is intentionally sequence-aware and batch-oriented: it
    consumes the full frame buffer, exports a temporary video, invokes the
    external ``external/phase_mag`` MATLAB code, reads the processed video
    back, and returns a new magnified frame buffer.
    """

    capabilities = StageCapabilities.batch(
        stateful=True,
        realtime_safe=False,
    )

    def __init__(
        self,
        magnification_factor: float = 20.0,
        low_cutoff_hz: float = 0.4,
        high_cutoff_hz: float = 3.0,
        sampling_rate_hz: float | None = None,
        sigma: float = 0.0,
        pyr_type: str = "halfOctave",
        attenuate_other_frequencies: bool = False,
        scale_video: float = 1.0,
        use_matlab_runtime: str = "auto",
        executable: str | None = None,
        release_dir: str | Path | None = None,
        temp_dir: str | Path | None = None,
        fps: float | None = None,
        codec: str = _DEFAULT_CODEC,
        timeout_seconds: float = 600.0,
        keep_temp_files: bool = False,
        emit_intermediate_artifacts: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        merged_config = dict(config or {})
        resolved_release_dir = (
            Path(release_dir)
            if release_dir is not None
            else Path(__file__).resolve().parents[2] / "external" / "phase_mag" / "Release"
        )
        resolved_temp_dir = Path(temp_dir) if temp_dir is not None else None
        runtime = str(use_matlab_runtime).strip().lower()
        if runtime not in {"auto", "matlab", "octave"}:
            raise ValueError("use_matlab_runtime must be one of: auto, matlab, octave.")
        if magnification_factor <= 0:
            raise ValueError("magnification_factor must be greater than 0.")
        if low_cutoff_hz < 0 or high_cutoff_hz <= 0 or low_cutoff_hz >= high_cutoff_hz:
            raise ValueError("Expected 0 <= low_cutoff_hz < high_cutoff_hz.")
        if sampling_rate_hz is not None and sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be greater than 0 when provided.")
        if sigma < 0:
            raise ValueError("sigma must be greater than or equal to 0.")
        if scale_video <= 0:
            raise ValueError("scale_video must be greater than 0.")
        if fps is not None and fps <= 0:
            raise ValueError("fps must be greater than 0 when provided.")
        if len(codec) != 4:
            raise ValueError("codec must be a four-character OpenCV codec.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0.")

        merged_config.update(
            {
                "magnification_factor": magnification_factor,
                "low_cutoff_hz": low_cutoff_hz,
                "high_cutoff_hz": high_cutoff_hz,
                "sampling_rate_hz": sampling_rate_hz,
                "sigma": sigma,
                "pyr_type": pyr_type,
                "attenuate_other_frequencies": attenuate_other_frequencies,
                "scale_video": scale_video,
                "use_matlab_runtime": runtime,
                "executable": executable,
                "release_dir": str(resolved_release_dir),
                "temp_dir": str(resolved_temp_dir) if resolved_temp_dir is not None else None,
                "fps": fps,
                "codec": codec,
                "timeout_seconds": timeout_seconds,
                "keep_temp_files": keep_temp_files,
                "emit_intermediate_artifacts": emit_intermediate_artifacts,
            }
        )
        super().__init__(merged_config)
        self.magnification_factor = float(magnification_factor)
        self.low_cutoff_hz = float(low_cutoff_hz)
        self.high_cutoff_hz = float(high_cutoff_hz)
        self.sampling_rate_hz = float(sampling_rate_hz) if sampling_rate_hz is not None else None
        self.sigma = float(sigma)
        self.pyr_type = str(pyr_type)
        self.attenuate_other_frequencies = bool(attenuate_other_frequencies)
        self.scale_video = float(scale_video)
        self.use_matlab_runtime = runtime
        self.executable = executable
        self.release_dir = resolved_release_dir
        self.temp_dir = resolved_temp_dir
        self.fps = float(fps) if fps is not None else None
        self.codec = codec
        self.timeout_seconds = float(timeout_seconds)
        self.keep_temp_files = bool(keep_temp_files)
        self.emit_intermediate_artifacts = bool(emit_intermediate_artifacts)

    def process(self, buffer: FrameBuffer) -> FrameBuffer:
        return self._process(buffer, context=None)

    def process_with_context(
        self,
        buffer: FrameBuffer,
        context: FrameProcessorExecutionContext,
    ) -> FrameBuffer:
        return self._process(buffer, context=context)

    def _process(
        self,
        buffer: FrameBuffer,
        *,
        context: FrameProcessorExecutionContext | None,
    ) -> FrameBuffer:
        frames = self._read_and_validate_frames(buffer)
        fps = self._resolve_fps(frames)
        sampling_rate = self.sampling_rate_hz or fps
        log.info(
            "Phase magnification: processing %s frames at %.3f fps with sampling rate %.3f Hz.",
            len(frames),
            fps,
            sampling_rate,
        )
        output_frames = self._magnify_frames(frames, fps=fps, sampling_rate=sampling_rate)
        return self._build_output_buffer(frames, output_frames, fps=fps, sampling_rate=sampling_rate, context=context, capacity=buffer.capacity)

    def _read_and_validate_frames(self, buffer: FrameBuffer) -> list[Frame]:
        frames: list[Frame] = []
        expected_shape: tuple[int, ...] | None = None
        for frame in buffer:
            image = np.asarray(frame.image)
            if image.ndim not in (2, 3):
                raise ValueError(f"Phase magnification expects 2D or 3D images; got shape {image.shape}.")
            if image.size == 0:
                raise ValueError("Phase magnification cannot process empty frames.")
            if expected_shape is None:
                expected_shape = tuple(int(value) for value in image.shape)
            elif tuple(image.shape) != expected_shape:
                raise ValueError(
                    f"All frames must share the same shape for phase magnification. Expected {expected_shape}, got {image.shape}."
                )
            frames.append(frame)

        if not frames:
            raise ValueError("PhaseMagnificationFrameProcessor requires at least one frame.")
        return frames

    def _resolve_fps(self, frames: list[Frame]) -> float:
        if self.fps is not None:
            return self.fps

        for frame in frames:
            source_fps = frame.metadata.get("source_fps")
            if isinstance(source_fps, (int, float)) and float(source_fps) > 0:
                return float(source_fps)

        timestamps = [frame.timestamp_seconds for frame in frames if frame.timestamp_seconds is not None]
        if len(timestamps) >= 2:
            deltas = [
                float(current - previous)
                for previous, current in zip(timestamps, timestamps[1:])
                if current is not None and previous is not None and float(current - previous) > 0
            ]
            if deltas:
                return 1.0 / float(np.median(deltas))

        raise ValueError(
            "Cannot infer fps for phase magnification. Provide fps explicitly or ensure frame metadata includes source_fps."
        )

    def _magnify_frames(self, frames: list[Frame], *, fps: float, sampling_rate: float) -> list[np.ndarray]:
        temp_root_context = (
            tempfile.TemporaryDirectory(prefix="sef_phase_mag_", dir=self.temp_dir)
            if not self.keep_temp_files
            else _PersistentTempDir(prefix="sef_phase_mag_", root=self.temp_dir)
        )
        with temp_root_context as temp_root_name:
            temp_root = Path(temp_root_name)
            input_video = temp_root / _DEFAULT_INPUT_NAME
            output_video = temp_root / _DEFAULT_OUTPUT_NAME
            script_path = temp_root / "run_phase_mag.m"
            self._write_temp_video(input_video, frames, fps=fps)
            script_path.write_text(
                self._matlab_script(
                    input_video=input_video,
                    output_video=output_video,
                    sampling_rate=sampling_rate,
                    frame_count=len(frames),
                ),
                encoding="utf-8",
            )
            self._run_phase_magnification(script_path=script_path)
            return self._read_output_video(output_video, expected_count=len(frames))

    def _run_phase_magnification(self, *, script_path: Path) -> None:
        runtime, executable = self._resolve_runtime()
        command = self._command_for_runtime(runtime, executable, script_path)
        log.info("Phase magnification: invoking %s.", " ".join(command))
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Phase magnification runtime not found: {executable}") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = stderr or stdout or str(exc)
            raise RuntimeError(f"Phase magnification command failed: {details}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Phase magnification exceeded timeout of {self.timeout_seconds:.1f} seconds."
            ) from exc

    def _resolve_runtime(self) -> tuple[str, str]:
        if self.executable:
            return self.use_matlab_runtime if self.use_matlab_runtime != "auto" else "matlab", self.executable

        if self.use_matlab_runtime in {"auto", "matlab"}:
            matlab = shutil.which("matlab")
            if matlab is not None:
                return "matlab", matlab
        if self.use_matlab_runtime in {"auto", "octave"}:
            octave = shutil.which("octave") or shutil.which("octave-cli")
            if octave is not None:
                return "octave", octave

        expected = "matlab or octave" if self.use_matlab_runtime == "auto" else self.use_matlab_runtime
        raise RuntimeError(f"Phase magnification runtime '{expected}' was not found on PATH.")

    @staticmethod
    def _command_for_runtime(runtime: str, executable: str, script_path: Path) -> list[str]:
        script = script_path.as_posix()
        if runtime == "matlab":
            return [executable, "-batch", f"run('{script}')"]
        return [executable, "--quiet", "--eval", f"run('{script}')"]

    def _matlab_script(
        self,
        *,
        input_video: Path,
        output_video: Path,
        sampling_rate: float,
        frame_count: int,
    ) -> str:
        release_dir = self.release_dir.resolve()
        if not release_dir.exists():
            raise RuntimeError(f"Phase magnification release directory not found: {release_dir}")
        escaped_release = self._matlab_string(release_dir)
        escaped_input = self._matlab_string(input_video)
        escaped_output = self._matlab_string(output_video)
        escaped_out_dir = self._matlab_string(output_video.parent)
        pyr_type = self._matlab_string(self.pyr_type)
        attenuate = "true" if self.attenuate_other_frequencies else "false"
        return "\n".join(
            [
                f"cd('{escaped_release}');",
                "setPath;",
                "try",
                "    if exist('phaseAmplify', 'file') ~= 2",
                "        error('phaseAmplify function not found after setPath.');",
                "    end",
                f"    outName = phaseAmplify('{escaped_input}', {self.magnification_factor:.12g}, {self.low_cutoff_hz:.12g}, {self.high_cutoff_hz:.12g}, {sampling_rate:.12g}, '{escaped_out_dir}', ...",
                f"        'attenuateOtherFreq', {attenuate}, 'pyrType', '{pyr_type}', 'sigma', {self.sigma:.12g}, 'scaleVideo', {self.scale_video:.12g}, 'useFrames', [1 {frame_count}]);",
                "    sourceFile = fullfile('" + escaped_out_dir + "', outName);",
                "    if exist(sourceFile, 'file') ~= 2",
                "        error('Phase magnification output video was not created.');",
                "    end",
                f"    if exist('{escaped_output}', 'file') == 2",
                f"        delete('{escaped_output}');",
                "    end",
                f"    copyfile(sourceFile, '{escaped_output}');",
                "catch err",
                "    fprintf(2, '%s\\n', err.message);",
                "    rethrow(err);",
                "end",
            ]
        )

    @staticmethod
    def _matlab_string(value: str | Path) -> str:
        return str(value).replace("\\", "/").replace("'", "''")

    def _build_output_buffer(
        self,
        source_frames: list[Frame],
        output_frames: list[np.ndarray],
        *,
        fps: float,
        sampling_rate: float,
        context: FrameProcessorExecutionContext | None,
        capacity: int,
    ) -> FrameBuffer:
        if len(output_frames) != len(source_frames):
            raise RuntimeError(
                f"Phase magnification output frame count mismatch: expected {len(source_frames)}, got {len(output_frames)}."
            )

        output = FrameBuffer(buffer_size=max(len(output_frames) + 1, capacity))
        metadata = self._processor_metadata(fps=fps, sampling_rate=sampling_rate, frame_count=len(output_frames))
        for sequence_index, (source_frame, image) in enumerate(zip(source_frames, output_frames)):
            frame = Frame(
                image=image,
                index=source_frame.index,
                timestamp_seconds=source_frame.timestamp_seconds,
                metadata={
                    **dict(source_frame.metadata),
                    _METADATA_KEY: metadata,
                },
            )
            output.put(frame)
            self._capture_intermediate_artifact(
                original_frame=source_frame,
                processed_frame=frame,
                source_sequence_index=sequence_index,
                context=context,
            )
        output.close()
        return output

    def _capture_intermediate_artifact(
        self,
        *,
        original_frame: Frame,
        processed_frame: Frame,
        source_sequence_index: int,
        context: FrameProcessorExecutionContext | None,
    ) -> None:
        if context is None or context.intermediate_store is None:
            return
        if not context.intermediate_store.should_capture(source_sequence_index):
            return
        if not (self.emit_intermediate_artifacts or context.intermediate_store.config.enabled):
            return

        context.intermediate_store.add(
            IntermediateFrameArtifact(
                image=processed_frame.image,
                stage_name=context.stage_name,
                frame_index=processed_frame.index,
                timestamp_seconds=processed_frame.timestamp_seconds,
                original_image=original_frame.image if context.intermediate_store.config.include_original else None,
                processed_image=processed_frame.image,
                stage_metadata={
                    "source_sequence_index": source_sequence_index,
                    "processor_name": context.processor_name,
                },
                metadata=dict(processed_frame.metadata.get(_METADATA_KEY, {})),
                config=dict(self.config),
            ),
            source_sequence_index=source_sequence_index,
        )

    def _processor_metadata(self, *, fps: float, sampling_rate: float, frame_count: int) -> dict[str, Any]:
        return {
            "magnification_factor": self.magnification_factor,
            "low_cutoff_hz": self.low_cutoff_hz,
            "high_cutoff_hz": self.high_cutoff_hz,
            "sigma": self.sigma,
            "pyr_type": self.pyr_type,
            "attenuate_other_frequencies": self.attenuate_other_frequencies,
            "scale_video": self.scale_video,
            "fps": fps,
            "sampling_rate_hz": sampling_rate,
            "frame_count": frame_count,
            "runtime": self.use_matlab_runtime,
            "release_dir": str(self.release_dir),
        }

    def _write_temp_video(self, path: Path, frames: list[Frame], *, fps: float) -> None:
        writer: cv2.VideoWriter | None = None
        expected_size: tuple[int, int] | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            for frame in frames:
                image = self._video_image(frame.image)
                size = (int(image.shape[1]), int(image.shape[0]))
                if expected_size is None:
                    expected_size = size
                    writer = cv2.VideoWriter(
                        str(path),
                        cv2.VideoWriter_fourcc(*self.codec),
                        fps,
                        expected_size,
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create temporary phase magnification input video: {path}")
                elif size != expected_size:
                    raise ValueError(f"All frames must have the same size. Expected {expected_size}, got {size}.")
                writer.write(image)
        finally:
            if writer is not None:
                writer.release()
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"Temporary phase magnification input video was not created: {path}")

    def _read_output_video(self, path: Path, *, expected_count: int) -> list[np.ndarray]:
        if not path.exists():
            raise RuntimeError(f"Phase magnification output video not found: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open phase magnification output video: {path}")
        frames: list[np.ndarray] = []
        try:
            while True:
                success, image = capture.read()
                if not success:
                    break
                frames.append(np.ascontiguousarray(image))
        finally:
            capture.release()
        if len(frames) != expected_count:
            raise RuntimeError(
                f"Phase magnification output video frame count mismatch: expected {expected_count}, got {len(frames)}."
            )
        return frames

    @staticmethod
    def _video_image(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.dtype == np.bool_:
            image = image.astype(np.uint8) * 255
        elif np.issubdtype(image.dtype, np.floating):
            finite = image[np.isfinite(image)]
            if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
                image = np.clip(image, 0.0, 1.0) * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_GRAY2BGR)
        if image.ndim != 3:
            raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}.")
        channels = image.shape[2]
        if channels == 1:
            return cv2.cvtColor(np.ascontiguousarray(image[:, :, 0]), cv2.COLOR_GRAY2BGR)
        if channels == 3:
            return np.ascontiguousarray(image)
        if channels == 4:
            return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGRA2BGR)
        raise ValueError(f"Unsupported channel count for video export: {channels}.")


class _PersistentTempDir:
    """Small context manager mirroring TemporaryDirectory without automatic cleanup."""

    def __init__(self, *, prefix: str, root: Path | None) -> None:
        self._prefix = prefix
        self._root = root
        self._path: str | None = None

    def __enter__(self) -> str:
        self._path = tempfile.mkdtemp(prefix=self._prefix, dir=self._root)
        return self._path

    def __exit__(self, exc_type, exc, tb) -> None:
        return None
