import cv2
from typing import Tuple
from library.core.abstractions.ISignalExtractor import ISignalExtractor
from library.core.abstractions.ISignal import ISignal
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal


class OpenCVBufferedSignalExtractor(ISignalExtractor):

    def __init__(self, tracker_type: str = "CSRT", start_box: Tuple[int, int, int, int] = [0,0,0,0]):
        self.tracker = self._create_tracker(tracker_type)
        self.box = start_box

    def track(self, buffer: FrameBuffer) -> ISignal:
        results = []
        first_frame = True

        for frame_number, frame in enumerate(buffer):
            if first_frame:
                self.tracker.init(frame.frame, self.box)
                first_frame = False
            else:
                success, box = self.tracker.update(frame)
                if not success:
                    #TODO: aggiungere controllo nel caso di insuccesso
                    pass
                x, y, w, h = box
                self.box = int(x), int(y), int(w), int(h)

            if self.box:
                x, y, w, h = self.box
                cx = x + w // 2
                cy = y + h // 2
                results.append({
                    'frame_number': frame_number,
                    'box': (x, y, w, h),
                    'centroid': (cx, cy)
                })
            else:
                results.append({
                    'frame_number': frame_number,
                    'box': None,
                    'centroid': None
                })

        return Signal(results)


    @staticmethod
    def _create_tracker(tracker_type):
        #TODO: vedere se effettivamente funzionano tutti
        if tracker_type == "CSRT":
            return cv2.legacy.TrackerCSRT_create()
        elif tracker_type == "KCF":
            return cv2.legacy.TrackerKCF_create()
        elif tracker_type == "MIL":
            return cv2.legacy.TrackerMIL_create()
        elif tracker_type == "GOTURN":
            return cv2.TrackerGOTURN_create()
        else:
            raise ValueError(f"Tracker {tracker_type} non supportato")