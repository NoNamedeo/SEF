from __future__ import annotations

from dataclasses import dataclass

from sef.core.interfaces.IData import IData


@dataclass(slots=True)
class NoData(IData):
    """No Data."""
