import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from sef.core.artifacts.signal_sample.BoxSignalSample import BoxSignalSample
from sef.core.interfaces.ILiveAnalyzer import ILiveAnalyzer


class LiveVerticalPositionAnalyzer(ILiveAnalyzer):
    def __init__(self, max_points: int = 500):
        # App Qt (una sola per processo)
        super().__init__()
        self.app = pg.mkQApp("Realtime XY Plot")

        # Finestra e grafico
        self.win = pg.plot(title="Realtime XY")
        self.curve = self.win.plot(pen='y', symbol='o', symbolSize=5)

        # Dati
        self.x_data = []
        self.y_data = []
        self.max_points = max_points

    def update(self, sample: BoxSignalSample):
        """Aggiunge un punto e aggiorna il grafico"""

        if sample.centroid is None:
            return

        x_axis_value = float(sample.timestamp_seconds if sample.timestamp_seconds is not None else float(sample.frame_index))
        y_axis_value = float(-sample.centroid[1]) #aggiungo meno perchè openCV prende le y partendo dall'alto

        self.x_data.append(x_axis_value)
        self.y_data.append(y_axis_value)

        # limita dimensione buffer
        if len(self.x_data) > self.max_points:
            self.x_data.pop(0)
            self.y_data.pop(0)

        self.curve.setData(self.x_data, self.y_data)

        QtWidgets.QApplication.processEvents()

    def start(self):
        """Avvia la GUI (bloccante)"""
        self.app.exec()