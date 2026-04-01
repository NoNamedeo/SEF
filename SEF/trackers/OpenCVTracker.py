import cv2
from typing import Tuple, List, Dict, Optional
import numpy as np
from SEF.abstractions.ITracker import ITracker
from SEF.abstractions.IExtractor import IExtractor

class OpenCVTracker(ITracker):
    """
    Tracker ad alto livello che prende in input un extractor
    e itera internamente sui frame prodotti dal generator.
    """

    def __init__(self, tracker_type: str = "CSRT", config: Optional[dict] = None):
        self.tracker_type = tracker_type.upper()
        self.tracker: Optional[cv2.Tracker] = None
        self.config = config or {}

    def _create_tracker(self):
        """Crea il tracker OpenCV richiesto"""
        if self.tracker_type == "CSRT":
            return cv2.legacy.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            return cv2.legacy.TrackerKCF_create()
        elif self.tracker_type == "MIL":
            return cv2.legacy.TrackerMIL_create()
        elif self.tracker_type == "GOTURN":
            return cv2.TrackerGOTURN_create()
        else:
            raise ValueError(f"Tracker {self.tracker_type} non supportato")

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Inizializza il tracker su un frame"""
        self.tracker = self._create_tracker()
        self.tracker.init(frame, bbox)

    def update(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Aggiorna il tracker e ritorna la nuova bbox"""
        success, bbox = self.tracker.update(frame)
        if not success:
            return None
        x, y, w, h = bbox
        return int(x), int(y), int(w), int(h)

    def track(self, extractor: IExtractor, init_bbox: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Traccia l'oggetto usando un extractor.

        Parameters
        ----------
        extractor : IExtractor
            Oggetto che implementa il metodo extract() -> generator di frame
        init_bbox : Tuple[int, int, int, int]
            Bounding box iniziale (x, y, w, h)

        Returns
        -------
        List[Dict]
            Lista di dizionari per frame:
            {
                'frame_idx': int,
                'bbox': (x, y, w, h),
                'centroid': (cx, cy)
            }
        """
        results = []
        first_frame = True

        for frame_idx, frame in enumerate(extractor.extract()):
            if first_frame:
                self.init(frame, init_bbox)
                bbox = init_bbox
                first_frame = False
            else:
                bbox = self.update(frame)

            if bbox:
                x, y, w, h = bbox
                cx = x + w // 2
                cy = y + h // 2
                results.append({
                    'frame_idx': frame_idx,
                    'bbox': (x, y, w, h),
                    'centroid': (cx, cy)
                })
            else:
                results.append({
                    'frame_idx': frame_idx,
                    'bbox': None,
                    'centroid': None
                })

        return results