import cv2
import numpy as np

from SEF.abstractions.IAnalyzer import IAnalyzer


class OpenCVYTimeAnalyzer(IAnalyzer):
    """
    Stima una funzione y(t) a partire dai dati puliti dal cleaner.

    Input atteso:
        [
            {
                "frame_idx": int,
                "centroid": (x, y)
            },
            ...
        ]

    Output:
        {
            "function": callable,
            "formula": str,
            "slope": float,
            "intercept": float,
            "times": np.ndarray,
            "y_values": np.ndarray
        }
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.fps = float(self.config.get("fps", 30.0))

        if self.fps <= 0:
            raise ValueError("fps deve essere > 0")

    def analyze(self, signal):
        valid_points = [
            item for item in signal
            if item.get("centroid") is not None and item.get("frame_idx") is not None
        ]

        if not valid_points:
            raise ValueError("Nessun dato valido da analizzare")

        times = np.array(
            [item["frame_idx"] / self.fps for item in valid_points],
            dtype=np.float32,
        )
        # OpenCV usa un asse y verso il basso; lo invertiamo per ottenere
        # un grafico fisicamente piu' intuitivo, dove un oggetto che sale
        # ha y crescente nel tempo.
        y_values = np.array(
            [-item["centroid"][1] for item in valid_points],
            dtype=np.float32,
        )

        if len(valid_points) == 1:
            intercept = float(y_values[0])

            def constant_function(t):
                t_array = np.asarray(t, dtype=np.float32)
                return np.full_like(t_array, intercept, dtype=np.float32)

            return {
                "function": constant_function,
                "formula": f"y(t) = {intercept:.4f}",
                "slope": 0.0,
                "intercept": intercept,
                "times": times,
                "y_values": y_values,
            }

        fit_points = np.column_stack((times, y_values)).reshape(-1, 1, 2)
        line = cv2.fitLine(
            fit_points,
            distType=cv2.DIST_L2,
            param=0,
            reps=0.01,
            aeps=0.01,
        )
        vx, vy, x0, y0 = np.asarray(line, dtype=np.float32).reshape(-1)[:4]

        vx = float(vx)
        vy = float(vy)
        x0 = float(x0)
        y0 = float(y0)

        if abs(vx) < 1e-8:
            raise ValueError("Impossibile stimare y(t): asse temporale degenerato")

        slope = vy / vx
        intercept = y0 - slope * x0

        def y_function(t):
            t_array = np.asarray(t, dtype=np.float32)
            return slope * t_array + intercept

        sign = "+" if intercept >= 0 else "-"
        formula = f"y(t) = {slope:.4f} * t {sign} {abs(intercept):.4f}"

        return {
            "function": y_function,
            "formula": formula,
            "slope": float(slope),
            "intercept": float(intercept),
            "times": times,
            "y_values": y_values,
        }
