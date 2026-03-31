import cv2
from library.src.extractors.OpenCVVideoExtractor import OpenCVVideoExtractor
from library.src.trackers.OpenCVTracker import OpenCVTracker

def main():
    # --- Configurazione ---
    video_path = "../../videos/Traffic.mp4"
    resize = (640, 480)
    use_interactive_bbox = True  # Se True, selezioni la box col mouse

    # --- Crea extractor ---
    extractor = OpenCVVideoExtractor(config={
        "resize": resize,
        "gray": False,  # BGR
        "stride": 1,
        "max_frames": None
    })

    # --- Crea tracker ---
    tracker = OpenCVTracker(tracker_type="CSRT")

    # --- Estrai il primo frame ---
    first_frame = next(extractor.extract(video_path))

    # --- Bounding box iniziale ---
    if use_interactive_bbox:
        # Seleziona la ROI con il mouse
        init_bbox = cv2.selectROI("Seleziona oggetto da tracciare", first_frame, False, False)
        cv2.destroyWindow("Seleziona oggetto da tracciare")
    else:
        # Bounding box predefinita
        init_bbox = (300, 200, 80, 120)

    # --- Inizializza tracker sul primo frame ---
    tracker.init(first_frame, init_bbox)

    # --- Visualizzazione del tracking ---
    for frame_idx, frame in enumerate(extractor.extract(video_path)):
        # Skip primo frame già usato
        if frame_idx == 0:
            bbox = init_bbox
        else:
            bbox = tracker.update(frame)

        # Disegna la bounding box sul frame
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC per uscire
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()