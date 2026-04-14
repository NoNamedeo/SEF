from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from library.core.interfaces.IData import IData


@dataclass(slots=True)
class CategoryData(IData):
    """
    Generic data container for categories.
    """

    # total counts per category
    category_counts: dict[str, int]

    # mapping: track_id -> categories assigned
    track_categories: dict[int, list[str]]

    # list of all the categories
    categories: list[str]

    metadata: dict[str, Any]
