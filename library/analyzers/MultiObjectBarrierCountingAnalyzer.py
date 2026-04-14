from __future__ import annotations

from typing import Any

from library.core.artifacts.CategoryData import CategoryData
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.ISignal import ISignal


class MultiObjectBarrierCountingAnalyzer(IAnalyzer):
    """
    Each category == a barrier.

    Counts how many unique objects (track_id)
    cross each barrier at least once.
    """

    def __init__(
        self,
        barriers: dict[str, tuple[tuple[float, float], tuple[float, float]]],
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

        self.barriers = barriers  # category -> line segment (dict di nome e due estremi della linea)

        # posizione precedente (dei track)
        self._prev_positions: dict[int, tuple[float, float]] = {}

        # lista di barriere già attraversate (dai track)
        self._crossed: dict[int, set[str]] = {}

    def analyze(self, signal: ISignal) -> CategoryData:

        # nome barriera e numero di passaggi
        category_counts: dict[str, int] = {b: 0 for b in self.barriers}

        # track id e lista barriere passate
        track_categories: dict[int, list[str]] = {}

        # lista delle barriere
        categories = list(self.barriers.keys())

        for sample in signal:
            for track in sample.tracks:
                if track.centroid is None:
                    continue

                tid = track.track_id
                current = track.centroid

                if tid in self._prev_positions:
                    prev = self._prev_positions[tid]

                    for barrier_name, (a1, a2) in self.barriers.items():
                        if self._cross(prev, current, a1, a2):
                            if tid not in self._crossed:
                                self._crossed[tid] = set()

                            # considero solo se è il primo attraversamento
                            if barrier_name not in self._crossed[tid]:
                                self._crossed[tid].add(barrier_name)
                                print(
                                    "Crossed barrier",
                                    barrier_name,
                                )

                                category_counts[barrier_name] += 1

                                track_categories.setdefault(tid, []).append(barrier_name)

                self._prev_positions[tid] = current

        return CategoryData(
            category_counts=category_counts,
            track_categories=track_categories,
            categories=categories,
            metadata={
                "total_tracks": len(self._prev_positions),
                "total_crossings": sum(category_counts.values()),
            },
        )

    @staticmethod
    def _cross(p1, p2, a1, a2) -> bool:
        """
        Segment intersection test.
        """

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, a1, a2) != ccw(p2, a1, a2) and ccw(p1, p2, a1) != ccw(p1, p2, a2)
