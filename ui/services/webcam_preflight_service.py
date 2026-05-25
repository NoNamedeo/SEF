"""Preflight checks for webcam-based UI runs."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WebcamPreflightResult:
    """Outcome of a short camera-open/read probe."""

    ok: bool
    camera_index: int
    message: str
    details: str = ""


def webcam_camera_index(config: dict[str, Any]) -> int | None:
    """Return the configured webcam index, or None when the config is not webcam-based."""
    pipeline = config.get("pipeline", {})
    if not isinstance(pipeline, dict):
        return None
    frame_extractor = pipeline.get("frame_extractor", {})
    if not isinstance(frame_extractor, dict) or str(frame_extractor.get("name", "")) != "opencv_webcam":
        return None
    params = frame_extractor.get("params", {})
    if not isinstance(params, dict):
        return 0
    return int(params.get("camera_index", 0))


def check_webcam_access(camera_index: int, *, timeout_seconds: float = 8.0) -> WebcamPreflightResult:
    """
    Probe webcam access in a separate process before submitting async pipelines.

    On macOS, OpenCV may need to trigger/check AVFoundation camera permission
    from a process main thread. Running this probe before the pipeline worker
    avoids a silent async run that never publishes preview frames.
    """
    script = """
import cv2
import sys

camera_index = int(sys.argv[1])
capture = cv2.VideoCapture(camera_index)
try:
    if not capture.isOpened():
        print(f"Cannot open webcam index {camera_index}.", file=sys.stderr)
        raise SystemExit(2)
    ok, frame = capture.read()
    if not ok or frame is None:
        print(f"Cannot read from webcam index {camera_index}.", file=sys.stderr)
        raise SystemExit(3)
    print(f"OK {frame.shape[1]}x{frame.shape[0]}")
finally:
    capture.release()
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, str(camera_index)],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return WebcamPreflightResult(
            ok=False,
            camera_index=camera_index,
            message=f"Timeout apertura webcam index {camera_index}.",
            details=str(exc),
        )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    details = "\n".join(item for item in (stdout, stderr) if item)
    if completed.returncode == 0:
        return WebcamPreflightResult(
            ok=True,
            camera_index=camera_index,
            message=f"Webcam index {camera_index} pronta.",
            details=details,
        )

    permission_hint = ""
    if "not authorized" in stderr.lower() or "avfoundation" in stderr.lower():
        permission_hint = (
            " macOS non ha autorizzato il processo Python/Streamlit alla camera: "
            "abilita Camera per Terminal/VSCode/Python in Impostazioni di Sistema > Privacy e Sicurezza > Fotocamera, "
            "poi riavvia Streamlit."
        )
    return WebcamPreflightResult(
        ok=False,
        camera_index=camera_index,
        message=f"Webcam index {camera_index} non disponibile.{permission_hint}",
        details=details,
    )
