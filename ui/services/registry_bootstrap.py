"""
Bootstrap the PluginRegistry with all built-in SEF components.

Each registration is wrapped in a try/except so that a broken component
(e.g. one that still imports from the old `library.core.interfaces.*`
path) does not prevent the rest of the registry from loading.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# ── project-root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from library.core.plugins.PluginRegistry import PluginRegistry, PluginCategory  # noqa: E402

log = logging.getLogger(__name__)


def _try_register(registry: PluginRegistry, category, name: str, factory, description: str = "") -> None:
    """Register a plugin, silently skipping it if its module is broken."""
    try:
        registry.register(category, name, factory, description)
    except Exception as exc:
        log.warning("Plugin '%s/%s' could not be registered: %s", category, name, exc)


@st.cache_resource(show_spinner="Caricamento plugin registry…")
def get_registry() -> PluginRegistry:
    """Return the shared PluginRegistry (created once per Streamlit server process)."""
    registry = PluginRegistry()

    # ── Frame extractors ──────────────────────────────────────────────────────
    try:
        from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor

        _try_register(
            registry, PluginCategory.FRAME_EXTRACTOR, "opencv_buffered", OpenCVBufferedFrameExtractor, "Estrae frame da video con OpenCV."
        )
    except Exception as exc:
        log.warning("OpenCVBufferedFrameExtractor non disponibile: %s", exc)

    # ── Frame cleaners ────────────────────────────────────────────────────────
    try:
        from library.frame_cleaners.OpenCVGrayFrameCleaner import OpenCVGrayFrameCleaner

        _try_register(registry, PluginCategory.FRAME_CLEANER, "opencv_gray", OpenCVGrayFrameCleaner, "Converte i frame in scala di grigi.")
    except Exception as exc:
        log.warning("OpenCVGrayFrameCleaner non disponibile: %s", exc)

    try:
        from library.frame_cleaners.SmoothingFrameCleaner import SmoothingFrameCleaner

        _try_register(registry, PluginCategory.FRAME_CLEANER, "smoothing", SmoothingFrameCleaner, "Smoothing temporale tra frame consecutivi.")
    except Exception as exc:
        log.warning("SmoothingFrameCleaner non disponibile: %s", exc)

    # ── Signal extractors ─────────────────────────────────────────────────────
    try:
        from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor

        _try_register(
            registry, PluginCategory.SIGNAL_EXTRACTOR, "opencv_tracker", OpenCVBufferedSignalExtractor, "Tracker singolo oggetto (CSRT/KCF/MIL)."
        )
    except Exception as exc:
        log.warning("OpenCVBufferedSignalExtractor non disponibile: %s", exc)

    try:
        from library.signal_extractors.OpenCVMultiObjectSignalExtractor import OpenCVMultiObjectSignalExtractor

        _try_register(
            registry,
            PluginCategory.SIGNAL_EXTRACTOR,
            "opencv_multi_tracker",
            OpenCVMultiObjectSignalExtractor,
            "Tracker multi-oggetto con espansione automatica.",
        )
    except Exception as exc:
        log.warning("OpenCVMultiObjectSignalExtractor non disponibile: %s", exc)

    # ── Signal cleaners ───────────────────────────────────────────────────────
    try:
        from library.signal_cleaners.MovingAverageCleaner import MovingAverageCleaner

        _try_register(registry, PluginCategory.SIGNAL_CLEANER, "moving_average", MovingAverageCleaner, "Smoothing centroidi con media mobile.")
    except Exception as exc:
        log.warning("MovingAverageCleaner non disponibile: %s", exc)

    try:
        from library.signal_cleaners.OutlierRejectionCleaner import OutlierRejectionCleaner

        _try_register(registry, PluginCategory.SIGNAL_CLEANER, "outlier_rejection", OutlierRejectionCleaner, "Rimozione outlier dal segnale.")
    except Exception as exc:
        log.warning("OutlierRejectionCleaner non disponibile: %s", exc)

    # ── Analyzers ─────────────────────────────────────────────────────────────
    try:
        from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer

        _try_register(
            registry, PluginCategory.ANALYZER, "vertical_position", VerticalPositionAnalyzer, "Serie temporale della posizione verticale."
        )
    except Exception as exc:
        log.warning("VerticalPositionAnalyzer non disponibile: %s", exc)

    try:
        from library.analyzers.VerticalFrequencyAnalyzer import VerticalFrequencyAnalyzer

        _try_register(registry, PluginCategory.ANALYZER, "vertical_frequency", VerticalFrequencyAnalyzer, "Spettro di frequenza verticale (FFT).")
    except Exception as exc:
        log.warning("VerticalFrequencyAnalyzer non disponibile: %s", exc)

    try:
        from library.analyzers.HoriziontalPositionAnalyzer import HorizontalPositionAnalyzer  # noqa: N813

        _try_register(
            registry, PluginCategory.ANALYZER, "horizontal_position", HorizontalPositionAnalyzer, "Serie temporale della posizione orizzontale."
        )
    except Exception as exc:
        log.warning("HorizontalPositionAnalyzer non disponibile: %s", exc)

    try:
        from library.analyzers.VerticalVelocityAnalyzer import VerticalVelocityAnalyzer

        _try_register(registry, PluginCategory.ANALYZER, "vertical_velocity", VerticalVelocityAnalyzer, "Serie temporale della velocità verticale.")
    except Exception as exc:
        log.warning("VerticalVelocityAnalyzer non disponibile: %s", exc)

    try:
        from library.analyzers.HorizontalVelocityAnalyzer import HorizontalVelocityAnalyzer

        _try_register(
            registry, PluginCategory.ANALYZER, "horizontal_velocity", HorizontalVelocityAnalyzer, "Serie temporale della velocità orizzontale."
        )
    except Exception as exc:
        log.warning("HorizontalVelocityAnalyzer non disponibile: %s", exc)

    try:
        from library.analyzers.HorizontalFrequencyAnalyzer import HorizontalFrequencyAnalyzer

        _try_register(
            registry, PluginCategory.ANALYZER, "horizontal_frequency", HorizontalFrequencyAnalyzer, "Spettro di frequenza orizzontale (FFT)."
        )
    except Exception as exc:
        log.warning("HorizontalFrequencyAnalyzer non disponibile: %s", exc)

    try:
        from library.analyzers.MultiObjectBarrierCountingAnalyzer import MultiObjectBarrierCountingAnalyzer

        _try_register(
            registry,
            PluginCategory.ANALYZER,
            "barrier_counting",
            MultiObjectBarrierCountingAnalyzer,
            "Conteggio attraversamenti barriere (multi-oggetto).",
        )
    except Exception as exc:
        log.warning("MultiObjectBarrierCountingAnalyzer non disponibile: %s", exc)

    # ── Visualizers ───────────────────────────────────────────────────────────
    try:
        from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer

        _try_register(registry, PluginCategory.VISUALIZER, "matplotlib_function", MatplotlibFunctionVisualizer, "Grafico linea/scatter Matplotlib.")
    except Exception as exc:
        log.warning("MatplotlibFunctionVisualizer non disponibile: %s", exc)

    try:
        from library.visualizers.MatplotlibHistogramVisualizer import MatplotlibHistogramVisualizer

        _try_register(
            registry, PluginCategory.VISUALIZER, "matplotlib_histogram", MatplotlibHistogramVisualizer, "Istogramma/bar chart Matplotlib."
        )
    except Exception as exc:
        log.warning("MatplotlibHistogramVisualizer non disponibile: %s", exc)

    try:
        from library.visualizers.MatplotlibTrajectoryVisualizer import MatplotlibTrajectoryVisualizer

        _try_register(
            registry, PluginCategory.VISUALIZER, "matplotlib_trajectory", MatplotlibTrajectoryVisualizer, "Visualizzazione traiettoria Matplotlib."
        )
    except Exception as exc:
        log.warning("MatplotlibTrajectoryVisualizer non disponibile: %s", exc)

    loaded = len(registry.list())
    log.info("PluginRegistry pronto: %d plugin caricati.", loaded)
    return registry
