from __future__ import annotations

from library.core.artifacts.NoData import NoData
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.ISignal import ISignal


class NoAnalyzer(IAnalyzer):
    """No operation analyzer."""

    def __init__(self, config=None):
        super().__init__(config)

    def analyze(self, signal: ISignal) -> IData:
        return NoData()
