"""Local MJPEG server for Streamlit-independent realtime previews."""

from __future__ import annotations

import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlparse

import cv2
import numpy as np

from ui.services.realtime_preview_service import snapshot_for_id

log = logging.getLogger(__name__)

_BOUNDARY = "sef-realtime-frame"
_SERVER_LOCK = threading.Lock()
_SERVER: ThreadingHTTPServer | None = None
_SERVER_THREAD: threading.Thread | None = None


def mjpeg_stream_url(sink_id: str) -> str:
    """Return a local browser URL that streams the latest frames for a sink."""
    server = _ensure_server()
    encoded_sink_id = quote(sink_id, safe="")
    return f"http://127.0.0.1:{server.server_port}/stream/{encoded_sink_id}"


def _ensure_server() -> ThreadingHTTPServer:
    global _SERVER, _SERVER_THREAD
    with _SERVER_LOCK:
        if _SERVER is not None:
            return _SERVER

        _SERVER = _RealtimePreviewServer(("127.0.0.1", 0), _RealtimePreviewHandler)
        _SERVER_THREAD = threading.Thread(
            target=_SERVER.serve_forever,
            name="sef-realtime-mjpeg",
            daemon=True,
        )
        _SERVER_THREAD.start()
        log.info("Realtime MJPEG preview server listening on 127.0.0.1:%s.", _SERVER.server_port)
        return _SERVER


class _RealtimePreviewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _RealtimePreviewHandler(BaseHTTPRequestHandler):
    server_version = "SEFRealtimePreview/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/stream/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        sink_id = unquote(parsed.path.removeprefix("/stream/"))
        if not sink_id:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing sink id.")
            return
        self._stream_sink(sink_id)

    def log_message(self, format: str, *args) -> None:
        log.debug("MJPEG preview: " + format, *args)

    def _stream_sink(self, sink_id: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.end_headers()

        last_version = -1
        while True:
            snapshot = snapshot_for_id(sink_id)
            frame = snapshot.frame
            version = snapshot.version
            if frame is None:
                image = _placeholder("Waiting for first frame")
                version = -2
            elif version == last_version and snapshot.active:
                time.sleep(0.05)
                continue
            else:
                image = frame.as_rgb()

            payload = _encode_jpeg(image)
            try:
                self.wfile.write(f"--{_BOUNDARY}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionError, OSError):
                return

            last_version = version
            time.sleep(0.1 if snapshot.active else 0.5)


def _encode_jpeg(image_rgb: np.ndarray) -> bytes:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("Could not encode realtime preview frame as JPEG.")
    return encoded.tobytes()


def _placeholder(message: str) -> np.ndarray:
    image = np.full((360, 640, 3), (28, 31, 36), dtype=np.uint8)
    cv2.putText(
        image,
        message,
        (32, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (230, 235, 240),
        2,
        cv2.LINE_AA,
    )
    return image
