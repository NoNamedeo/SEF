import cv2
import os
from pathlib import Path

from SEF.analyzers.OpenCVYTimeAnalyzer import OpenCVYTimeAnalyzer
from SEF.cleaners.OpenCVMovingAverageCleaner import OpenCVMovingAverageCleaner
from SEF.extractors.OpenCVVideoExtractor import OpenCVVideoExtractor
from SEF.trackers.OpenCVTracker import OpenCVTracker
from SEF.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer


def main():
    # --- Configurazione ---
    '''
    TODO: non mi funziona, se quello messo sotto funziona anche a te, togli questa parte

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    video_path = os.path.join(root_dir, "videos", "Crowd.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video NON trovato: {video_path}")
    '''

    root_dir = Path(__file__).resolve().parents[1]
    video_path = root_dir / "videos" / "Crowd.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video NON trovato: {video_path}")

    resize = (640, 480)
    fps = 30.0
    use_interactive_bbox = True  # Se True, selezioni la box col mouse

    # --- Crea i componenti della pipeline ---
    extractor = OpenCVVideoExtractor(config={
        "resize": resize,
        "gray": False,
        "stride": 5,
        "max_frames": None,
    })
    tracker = OpenCVTracker(tracker_type="CSRT")
    cleaner = OpenCVMovingAverageCleaner(config={"window_size": 5})
    analyzer = OpenCVYTimeAnalyzer(config={"fps": fps})
    visualizer = MatplotlibFunctionVisualizer()

    # --- Estrai il primo frame ---
    first_frame = next(extractor.extract(video_path))

    # --- Bounding box iniziale ---
    if use_interactive_bbox:
        init_bbox = cv2.selectROI("Seleziona oggetto da tracciare", first_frame, False, False)
        cv2.destroyWindow("Seleziona oggetto da tracciare")
    else:
        init_bbox = (300, 200, 80, 120)

    # --- Inizializza il tracker ---
    tracker.init(first_frame, init_bbox)

    # --- Tracking frame-by-frame ---
    tracking_results = []
    for frame_idx, frame in enumerate(extractor.extract(video_path)):
        if frame_idx == 0:
            bbox = init_bbox
        else:
            bbox = tracker.update(frame)

        if bbox:
            x, y, w, h = bbox
            centroid = (x + w // 2, y + h // 2)

            tracking_results.append({
                "frame_idx": frame_idx,
                "bbox": (x, y, w, h),
                "centroid": centroid,
            })

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Frame {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        else:
            tracking_results.append({
                "frame_idx": frame_idx,
                "bbox": None,
                "centroid": None,
            })

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    # --- Pipeline finale ---
    raw_analysis = analyzer.analyze(tracking_results)
    cleaned_results = cleaner.clean(tracking_results)
    cleaned_analysis = analyzer.analyze(cleaned_results)

    print(f"Dati grezzi: {raw_analysis['formula']}")
    visualizer.visualize(
        raw_analysis,
        title="Andamento di y nel tempo - dati grezzi",
        scatter_label="Centroidi grezzi",
        line_label="Fit sui dati grezzi",
    )

    print(f"Dati puliti: {cleaned_analysis['formula']}")
    visualizer.visualize(
        cleaned_analysis,
        title="Andamento di y nel tempo - dati puliti",
        scatter_label="Centroidi puliti",
        line_label="Fit sui dati puliti",
    )


if __name__ == "__main__":
    main()
