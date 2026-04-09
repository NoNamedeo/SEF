from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from library.services import (
    TrackingAnalysisConfig,
    build_analysis_csv,
    load_first_frame,
    run_tracking_analysis,
)

DEMO_VIDEO_DIR = PROJECT_ROOT / "videos"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "sef_streamlit_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False)
def get_demo_video_paths() -> list[str]:
    extensions = {".mp4", ".mov", ".avi", ".mkv"}
    return sorted(
        str(path)
        for path in DEMO_VIDEO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


@st.cache_data(show_spinner=False)
def get_first_frame(video_path: str) -> tuple:
    return load_first_frame(video_path)


@st.cache_data(show_spinner=False)
def read_video_bytes(video_path: str) -> bytes:
    return Path(video_path).read_bytes()


def save_uploaded_video(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    file_hash = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()[:16]
    target_path = UPLOAD_DIR / f"{file_hash}{suffix}"
    if not target_path.exists():
        target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def compute_resize(original_width: int, original_height: int, resize_option: str):
    if resize_option == "Originale":
        return None

    target_width = int(resize_option)
    target_height = max(1, int(round(original_height * (target_width / original_width))))
    return target_width, target_height


def default_bbox(frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    box_width = max(40, frame_width // 4)
    box_height = max(40, frame_height // 4)
    x = max(0, (frame_width - box_width) // 2)
    y = max(0, (frame_height - box_height) // 2)
    return x, y, box_width, box_height


def draw_bbox_preview(frame_rgb, bbox):
    preview = frame_rgb.copy()
    x, y, w, h = bbox
    cv2.rectangle(preview, (x, y), (x + w, y + h), (64, 214, 124), 2)
    cv2.circle(preview, (x + w // 2, y + h // 2), 4, (255, 210, 48), -1)
    return preview


def main():
    st.set_page_config(
        page_title="SEF Streamlit UI",
        layout="wide",
    )

    st.title("SEF Video Analysis Studio")
    st.caption(
        "Interfaccia Streamlit per caricare un video, impostare la ROI iniziale "
        "ed eseguire l'analisi con preview live, grafici e tabella dei segnali."
    )

    with st.sidebar:
        st.header("Configurazione")
        source_mode = st.radio("Sorgente video", ("Demo libreria", "Upload"))

        uploaded_file = None
        selected_demo_path = None
        if source_mode == "Demo libreria":
            demo_paths = get_demo_video_paths()
            if demo_paths:
                selected_demo_path = st.selectbox(
                    "Video di esempio",
                    demo_paths,
                    format_func=lambda path: Path(path).name,
                )
            else:
                st.warning("Nessun video demo disponibile nella cartella videos/.")
        else:
            uploaded_file = st.file_uploader(
                "Carica un video",
                type=["mp4", "mov", "avi", "mkv"],
            )

        tracker_type = st.selectbox("Tracker OpenCV", ("CSRT", "KCF", "MIL"))
        resize_option = st.selectbox("Resize analisi", ("Originale", "480", "640", "960"))
        stride = st.slider("Campionamento frame", min_value=1, max_value=10, value=1)
        smoothing_window = st.slider(
            "Finestra smoothing",
            min_value=1,
            max_value=21,
            value=5,
            step=2,
        )
        preview_stride = st.slider(
            "Aggiornamento preview live",
            min_value=1,
            max_value=20,
            value=3,
        )
        limit_frames = st.checkbox("Limita numero di frame", value=False)
        max_frames = None
        if limit_frames:
            max_frames = st.number_input(
                "Max frame da processare",
                min_value=10,
                max_value=5000,
                value=300,
                step=10,
            )

    video_path = None
    video_bytes = None
    if source_mode == "Demo libreria" and selected_demo_path:
        video_path = Path(selected_demo_path)
        video_bytes = read_video_bytes(str(video_path))
    elif uploaded_file is not None:
        video_path = save_uploaded_video(uploaded_file)
        video_bytes = uploaded_file.getvalue()

    if video_path is None:
        st.info("Seleziona un video demo oppure caricane uno per iniziare.")
        return

    first_frame_rgb, metadata = get_first_frame(str(video_path))
    resize_shape = compute_resize(metadata.width, metadata.height, resize_option)
    display_frame = first_frame_rgb
    if resize_shape is not None:
        display_frame = cv2.resize(first_frame_rgb, resize_shape)

    frame_height, frame_width = display_frame.shape[:2]
    video_signature = f"{video_path}:{frame_width}x{frame_height}"
    if st.session_state.get("bbox_signature") != video_signature:
        x, y, w, h = default_bbox(frame_width, frame_height)
        st.session_state["bbox_signature"] = video_signature
        st.session_state["bbox_x"] = x
        st.session_state["bbox_y"] = y
        st.session_state["bbox_w"] = w
        st.session_state["bbox_h"] = h

    st.subheader("Input video e ROI iniziale")
    preview_col, player_col = st.columns((1.2, 1), gap="large")

    with preview_col:
        st.markdown("**ROI iniziale**")
        x = st.slider("X", 0, max(frame_width - 1, 0), key="bbox_x")
        y = st.slider("Y", 0, max(frame_height - 1, 0), key="bbox_y")
        max_width = max(frame_width - x, 1)
        max_height = max(frame_height - y, 1)
        st.session_state["bbox_w"] = min(st.session_state["bbox_w"], max_width)
        st.session_state["bbox_h"] = min(st.session_state["bbox_h"], max_height)
        w = st.slider("Larghezza", 1, max_width, key="bbox_w")
        h = st.slider("Altezza", 1, max_height, key="bbox_h")
        bbox = (x, y, w, h)
        st.image(
            draw_bbox_preview(display_frame, bbox),
            channels="RGB",
            use_container_width=True,
            caption="Frame iniziale con ROI impostata",
        )

    with player_col:
        st.markdown("**Player sorgente**")
        if video_bytes is not None:
            st.video(video_bytes)
        metrics_cols = st.columns(3)
        metrics_cols[0].metric("Risoluzione", f"{metadata.width}x{metadata.height}")
        metrics_cols[1].metric("FPS", f"{metadata.fps:.2f}" if metadata.fps else "N/A")
        metrics_cols[2].metric(
            "Durata",
            f"{metadata.duration_seconds:.2f}s" if metadata.duration_seconds else "N/A",
        )

    st.subheader("Esecuzione")
    run_analysis = st.button("Avvia analisi", type="primary", use_container_width=True)

    current_result = None
    if run_analysis:
        live_status = st.empty()
        live_progress = st.progress(0.0, text="Preparazione analisi")
        live_preview = st.empty()

        def handle_preview(preview):
            live_status.markdown(
                f"**Frame processati:** {preview.processed_frames}  \n"
                f"**Frame corrente:** {preview.frame_index}"
            )
            live_progress.progress(
                preview.progress,
                text=f"Analisi in corso: frame {preview.frame_index}",
            )
            live_preview.image(
                preview.image_rgb,
                channels="RGB",
                use_container_width=True,
                caption=f"Preview live - frame {preview.frame_index}",
            )

        analysis_config = TrackingAnalysisConfig(
            video_path=str(video_path),
            bounding_box=bbox,
            tracker_type=tracker_type,
            resize=resize_shape,
            stride=stride,
            max_frames=int(max_frames) if max_frames is not None else None,
            smoothing_window=smoothing_window,
            preview_stride=preview_stride,
            fps_override=metadata.fps if metadata.fps > 0 else None,
        )

        try:
            current_result = run_tracking_analysis(
                config=analysis_config,
                on_preview=handle_preview,
            )
            st.session_state["analysis_result"] = current_result
            st.session_state["analysis_video_path"] = str(video_path)
            st.session_state["analysis_resize"] = resize_shape
            live_progress.progress(1.0, text="Analisi completata")
        except Exception as exc:
            live_progress.empty()
            live_preview.empty()
            st.error(f"Analisi fallita: {exc}")

    if current_result is None and st.session_state.get("analysis_video_path") == str(video_path):
        current_result = st.session_state.get("analysis_result")

    if current_result is None:
        return

    st.subheader("Risultati")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Frame processati", len(current_result.raw_signal))
    summary_cols[1].metric("Tracking riuscito", current_result.tracked_frames)
    summary_cols[2].metric("Tracking perso", current_result.lost_frames)
    summary_cols[3].metric("Success rate", f"{current_result.success_ratio:.1%}")

    if current_result.last_preview_frame is not None:
        st.image(
            current_result.last_preview_frame,
            channels="RGB",
            use_container_width=True,
            caption="Ultimo frame annotato elaborato",
        )

    chart_rows = current_result.chart_rows
    if chart_rows:
        chart_col_1, chart_col_2 = st.columns(2, gap="large")
        with chart_col_1:
            st.markdown("**Andamento asse X**")
            st.line_chart(
                chart_rows,
                x="time_seconds",
                y=["raw_x", "cleaned_x"],
                use_container_width=True,
            )
        with chart_col_2:
            st.markdown("**Andamento asse Y**")
            st.line_chart(
                chart_rows,
                x="time_seconds",
                y=["raw_y", "cleaned_y"],
                use_container_width=True,
            )

        st.markdown("**Segnali estratti**")
        st.dataframe(chart_rows, use_container_width=True)

        st.download_button(
            label="Scarica CSV segnali",
            data=build_analysis_csv(current_result),
            file_name=f"{video_path.stem}_signals.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
