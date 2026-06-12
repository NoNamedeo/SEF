"""
Bootstrap the PluginRegistry with all built-in SEF components.

Each registration is wrapped in a try/except so that a broken component
(e.g. one that still imports from the old `sef.core.interfaces.*`
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

from sef.core.plugins.PluginRegistry import PluginCategory, PluginRegistry  # noqa: E402

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
        from sef.builtin.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor

        _try_register(
            registry, PluginCategory.FRAME_EXTRACTOR, "opencv_buffered", OpenCVBufferedFrameExtractor, "Estrae frame da video con OpenCV."
        )
    except Exception as exc:
        log.warning("OpenCVBufferedFrameExtractor non disponibile: %s", exc)

    try:
        from sef.builtin.frame_extractors.OpenCVWebcamFrameExtractor import OpenCVWebcamFrameExtractor

        _try_register(
            registry,
            PluginCategory.FRAME_EXTRACTOR,
            "opencv_webcam",
            OpenCVWebcamFrameExtractor,
            "Acquisisce frame live da webcam con OpenCV.",
        )
    except Exception as exc:
        log.warning("OpenCVWebcamFrameExtractor non disponibile: %s", exc)

    # ── Frame processors ────────────────────────────────────────────────────────
    try:
        from sef.builtin.frame_processors.OpenCV.OpenCVGrayFrameProcessor import OpenCVGrayFrameProcessor

        _try_register(
            registry, PluginCategory.SINGLE_FRAME_PROCESSOR, "opencv_gray", OpenCVGrayFrameProcessor, "Converte i frame in scala di grigi."
        )
    except Exception as exc:
        log.warning("OpenCVGrayFrameProcessor non disponibile: %s", exc)

    try:
        from sef.builtin.frame_processors.SmoothingFrameProcessor import SmoothingFrameProcessor

        _try_register(
            registry, PluginCategory.SINGLE_FRAME_PROCESSOR, "smoothing", SmoothingFrameProcessor, "Smoothing temporale tra frame consecutivi."
        )
    except Exception as exc:
        log.warning("SmoothingFrameProcessor non disponibile: %s", exc)

    try:
        from sef.builtin.frame_processors.OpenCV.OpenCVResizeFrameProcessor import OpenCVResizeFrameProcessor

        _try_register(
            registry,
            PluginCategory.SINGLE_FRAME_PROCESSOR,
            "opencv_resize",
            OpenCVResizeFrameProcessor,
            "Ridimensiona i frame a una risoluzione fissa.",
        )
    except Exception as exc:
        log.warning("OpenCVResizeFrameProcessor non disponibile: %s", exc)

    try:
        from sef.builtin.frame_processors.OpenCV.OpenCVBackgroundSubtractionFrameProcessor import OpenCVBackgroundSubtractionFrameProcessor

        _try_register(
            registry,
            PluginCategory.SINGLE_FRAME_PROCESSOR,
            "background_subtraction",
            OpenCVBackgroundSubtractionFrameProcessor,
            "Isola oggetti in movimento tramite background subtraction.",
        )
    except Exception as exc:
        log.warning("OpenCVBackgroundSubtractionFrameProcessor non disponibile: %s", exc)

    try:
        from sef.builtin.frame_processors.OpenCV.OpenCVHistogramEqualizationFrameProcessor import OpenCVHistogramEqualizationFrameProcessor

        _try_register(
            registry,
            PluginCategory.SINGLE_FRAME_PROCESSOR,
            "histogram_equalization",
            OpenCVHistogramEqualizationFrameProcessor,
            "Migliora il contrasto dei frame tramite equalizzazione istogramma.",
        )
    except Exception as exc:
        log.warning("OpenCVHistogramEqualizationFrameProcessor non disponibile: %s", exc)

    try:
        from sef.builtin.frame_processors.ColorStabilizationFrameProcessor import ColorStabilizationFrameProcessor

        _try_register(
            registry,
            PluginCategory.SINGLE_FRAME_PROCESSOR,
            "color_stabilization",
            ColorStabilizationFrameProcessor,
            "Stabilizza luminosita, illuminazione e cromia tra frame consecutivi.",
        )
    except Exception as exc:
        log.warning("ColorStabilizationFrameProcessor non disponibile: %s", exc)

    try:
        from sef.builtin.frame_processors.DynamicObjectRemovalFrameProcessor import DynamicObjectRemovalFrameProcessor

        _try_register(
            registry,
            PluginCategory.FRAME_BUFFER_PROCESSOR,
            "dynamic_object_removal",
            DynamicObjectRemovalFrameProcessor,
            "Rimuove oggetti dinamici usando uno sfondo mediano temporale.",
        )
    except Exception as exc:
        log.warning("DynamicObjectRemovalFrameProcessor non disponibile: %s", exc)

    try:
        from sef.builtin.frame_processors.motion_magnification.PhaseMagnificationFrameProcessor import PhaseMagnificationFrameProcessor

        _try_register(
            registry,
            PluginCategory.FRAME_BUFFER_PROCESSOR,
            "motion_magnification",
            PhaseMagnificationFrameProcessor,
            "Amplifica micromovimenti tramite wrapper batch della pipeline phase-based esterna.",
        )
    except Exception as exc:
        log.warning("PhaseMagnificationFrameProcessor non disponibile: %s", exc)

    try:
        from sef.builtin.frame_processors.RealtimeFrameTapProcessor import RealtimeFrameTapProcessor
        from ui.services.realtime_preview_service import sink_for_id

        def realtime_frame_tap_factory(config=None, sink_id=None):
            return RealtimeFrameTapProcessor(config=config, sink=sink_for_id(sink_id))

        _try_register(
            registry,
            PluginCategory.FRAME_BUFFER_PROCESSOR,
            "realtime_frame_tap",
            realtime_frame_tap_factory,
            "Pubblica frame grezzi verso una preview realtime senza alterare la pipeline.",
        )
    except Exception as exc:
        log.warning("RealtimeFrameTapProcessor non disponibile: %s", exc)

    try:
        from sef.builtin.frame_processors.OpenCV.OpenCVInpaintFrameProcessor import OpenCVInpaintFrameProcessor

        _try_register(
            registry,
            PluginCategory.SINGLE_FRAME_PROCESSOR,
            "opencv_inpaint",
            OpenCVInpaintFrameProcessor,
            "Inpainting delle regioni mascherate con algoritmo Navier-Stokes o Telea.",
        )
    except Exception as exc:
        log.warning("OpenCVInpaintFrameProcessor non disponibile: %s", exc)

    try:
        from sef.builtin.frame_processors.OpenCV.OpenCVRotateFrameProcessor import OpenCVRotateFrameProcessor

        _try_register(
            registry,
            PluginCategory.SINGLE_FRAME_PROCESSOR,
            "opencv_rotate",
            OpenCVRotateFrameProcessor,
            "Ruota i frame di 0°, 90°, 180° o 270°.",
        )
    except Exception as exc:
        log.warning("OpenCVRotateFrameProcessor non disponibile: %s", exc)

    try:
        from sef.builtin.frame_processors.OpenCV.OpenCVZoomFrameProcessor import OpenCVZoomFrameProcessor

        _try_register(
            registry,
            PluginCategory.SINGLE_FRAME_PROCESSOR,
            "opencv_zoom",
            OpenCVZoomFrameProcessor,
            "Zoom digitale centrato sui frame.",
        )
    except Exception as exc:
        log.warning("OpenCVZoomFrameProcessor non disponibile: %s", exc)

    # ── Signal extractors ─────────────────────────────────────────────────────
    try:
        from sef.builtin.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor

        _try_register(
            registry, PluginCategory.SIGNAL_EXTRACTOR, "opencv_tracker", OpenCVBufferedSignalExtractor, "Tracker singolo oggetto (CSRT/KCF/MIL)."
        )
    except Exception as exc:
        log.warning("OpenCVBufferedSignalExtractor non disponibile: %s", exc)

    try:
        from sef.builtin.signal_extractors.OpenCVStreamSignalExtractor import OpenCVStreamSignalExtractor

        _try_register(
            registry,
            PluginCategory.SIGNAL_EXTRACTOR,
            "opencv_stream_tracker",
            OpenCVStreamSignalExtractor,
            "Tracker singolo oggetto con contratto streaming e preview live opzionale.",
        )
    except Exception as exc:
        log.warning("OpenCVStreamSignalExtractor non disponibile: %s", exc)

    try:
        from sef.builtin.signal_extractors.ArucoMarkerSignalExtractor import ArucoMarkerSignalExtractor

        _try_register(
            registry,
            PluginCategory.SIGNAL_EXTRACTOR,
            "aruco_marker",
            ArucoMarkerSignalExtractor,
            "Rileva marker ArUco configurabili frame-by-frame.",
        )
    except Exception as exc:
        log.warning("ArucoMarkerSignalExtractor non disponibile: %s", exc)

    try:
        from sef.builtin.signal_extractors.OpenCVMultiObjectSignalExtractor import OpenCVMultiObjectSignalExtractor

        _try_register(
            registry,
            PluginCategory.SIGNAL_EXTRACTOR,
            "opencv_multi_tracker",
            OpenCVMultiObjectSignalExtractor,
            "Tracker multi-oggetto con espansione automatica.",
        )
    except Exception as exc:
        log.warning("OpenCVMultiObjectSignalExtractor non disponibile: %s", exc)

    try:
        from sef.builtin.signal_extractors.OpenCVDenseOpticalFlowSignalExtractor import OpenCVDenseFarnebackSignalExtractor

        _try_register(
            registry,
            PluginCategory.SIGNAL_EXTRACTOR,
            "dense_optical_flow",
            OpenCVDenseFarnebackSignalExtractor,
            "Estrae un campo di moto denso con Farneback optical flow.",
        )
    except Exception as exc:
        log.warning("OpenCVDenseFarnebackSignalExtractor non disponibile: %s", exc)

    try:
        from sef.builtin.signal_extractors.YOLOSkeletonCOCOStreamSignalExtractor import YOLOSkeletonCOCOStreamSignalExtractor

        _try_register(
            registry,
            PluginCategory.SIGNAL_EXTRACTOR,
            "yolo_coco_pose",
            YOLOSkeletonCOCOStreamSignalExtractor,
            "Estrae keypoint COCO 17 in streaming con YOLO pose.",
        )
    except Exception as exc:
        log.warning("YOLOSkeletonCOCOStreamSignalExtractor non disponibile: %s", exc)

    # ── Signal cleaners ───────────────────────────────────────────────────────
    try:
        from sef.builtin.signal_cleaners.ArUco.ArucoTemporalStabilizerCleaner import ArucoTemporalStabilizerCleaner

        _try_register(
            registry,
            PluginCategory.SIGNAL_CLEANER,
            "aruco_temporal_stabilizer",
            ArucoTemporalStabilizerCleaner,
            "Stabilizza nel tempo center e corner dei marker ArUco con smoothing quality-aware.",
        )
    except Exception as exc:
        log.warning("ArucoTemporalStabilizerCleaner non disponibile: %s", exc)

    try:
        from sef.builtin.signal_cleaners.single_tracker.MovingAverageCleaner import MovingAverageCleaner

        _try_register(registry, PluginCategory.SIGNAL_CLEANER, "moving_average", MovingAverageCleaner, "Smoothing centroidi con media mobile.")
    except Exception as exc:
        log.warning("MovingAverageCleaner non disponibile: %s", exc)

    try:
        from sef.builtin.signal_cleaners.single_tracker.MovingAverageStreamSignalCleaner import MovingAverageStreamCleaner

        _try_register(
            registry,
            PluginCategory.SIGNAL_CLEANER,
            "moving_average_stream",
            MovingAverageStreamCleaner,
            "Smoothing causale dei centroidi compatibile con pipeline streaming.",
        )
    except Exception as exc:
        log.warning("MovingAverageStreamCleaner non disponibile: %s", exc)

    try:
        from sef.builtin.signal_cleaners.single_tracker.OutlierRejectionCleaner import OutlierRejectionCleaner

        _try_register(registry, PluginCategory.SIGNAL_CLEANER, "outlier_rejection", OutlierRejectionCleaner, "Rimozione outlier dal segnale.")
    except Exception as exc:
        log.warning("OutlierRejectionCleaner non disponibile: %s", exc)

    try:
        from sef.builtin.signal_cleaners.single_tracker.SignalWidenerCleaner import SignalWidenerCleaner

        _try_register(
            registry,
            PluginCategory.SIGNAL_CLEANER,
            "signal_widener",
            SignalWidenerCleaner,
            "Amplifica gli spostamenti del segnale rispetto alla media.",
        )
    except Exception as exc:
        log.warning("SignalWidenerCleaner non disponibile: %s", exc)

    try:
        from sef.builtin.signal_cleaners.COCO_pose.COCOSkeletonNormalizationSignalCleaner import COCOSkeletonNormalizationSignalCleaner

        _try_register(
            registry,
            PluginCategory.SIGNAL_CLEANER,
            "coco_skeleton_normalization",
            COCOSkeletonNormalizationSignalCleaner,
            "Normalizza keypoint COCO per analisi basate su skeleton.",
        )
    except Exception as exc:
        log.warning("COCOSkeletonNormalizationSignalCleaner non disponibile: %s", exc)

    # ── Analyzers ─────────────────────────────────────────────────────────────
    try:
        from sef.builtin.analyzers.single_tracker.VerticalPositionAnalyzer import VerticalPositionAnalyzer

        _try_register(
            registry, PluginCategory.ANALYZER, "vertical_position", VerticalPositionAnalyzer, "Serie temporale della posizione verticale."
        )
    except Exception as exc:
        log.warning("VerticalPositionAnalyzer non disponibile: %s", exc)

    try:
        from sef.builtin.analyzers.single_tracker.VerticalFrequencyAnalyzer import VerticalFrequencyAnalyzer

        _try_register(registry, PluginCategory.ANALYZER, "vertical_frequency", VerticalFrequencyAnalyzer, "Spettro di frequenza verticale (FFT).")
    except Exception as exc:
        log.warning("VerticalFrequencyAnalyzer non disponibile: %s", exc)

    try:
        from sef.builtin.analyzers.single_tracker.HoriziontalPositionAnalyzer import HorizontalPositionAnalyzer  # noqa: N813

        _try_register(
            registry, PluginCategory.ANALYZER, "horizontal_position", HorizontalPositionAnalyzer, "Serie temporale della posizione orizzontale."
        )
    except Exception as exc:
        log.warning("HorizontalPositionAnalyzer non disponibile: %s", exc)

    try:
        from sef.builtin.analyzers.single_tracker.VerticalVelocityAnalyzer import VerticalVelocityAnalyzer

        _try_register(registry, PluginCategory.ANALYZER, "vertical_velocity", VerticalVelocityAnalyzer, "Serie temporale della velocità verticale.")
    except Exception as exc:
        log.warning("VerticalVelocityAnalyzer non disponibile: %s", exc)

    try:
        from sef.builtin.analyzers.single_tracker.HorizontalVelocityAnalyzer import HorizontalVelocityAnalyzer

        _try_register(
            registry, PluginCategory.ANALYZER, "horizontal_velocity", HorizontalVelocityAnalyzer, "Serie temporale della velocità orizzontale."
        )
    except Exception as exc:
        log.warning("HorizontalVelocityAnalyzer non disponibile: %s", exc)

    try:
        from sef.builtin.analyzers.single_tracker.HorizontalFrequencyAnalyzer import HorizontalFrequencyAnalyzer

        _try_register(
            registry, PluginCategory.ANALYZER, "horizontal_frequency", HorizontalFrequencyAnalyzer, "Spettro di frequenza orizzontale (FFT)."
        )
    except Exception as exc:
        log.warning("HorizontalFrequencyAnalyzer non disponibile: %s", exc)

    try:
        from sef.builtin.analyzers.multiple_tracker.MultiObjectBarrierCountingAnalyzer import MultiObjectBarrierCountingAnalyzer

        _try_register(
            registry,
            PluginCategory.ANALYZER,
            "barrier_counting",
            MultiObjectBarrierCountingAnalyzer,
            "Conteggio attraversamenti barriere (multi-oggetto).",
        )
    except Exception as exc:
        log.warning("MultiObjectBarrierCountingAnalyzer non disponibile: %s", exc)

    try:
        from sef.builtin.analyzers.optical_flow.DenseOpticalFlowVectorFieldAnalyzer import DenseOpticalFlowVectorFieldAnalyzer

        _try_register(
            registry,
            PluginCategory.ANALYZER,
            "dense_vector_field",
            DenseOpticalFlowVectorFieldAnalyzer,
            "Costruisce dati vettoriali da dense optical flow.",
        )
    except Exception as exc:
        log.warning("DenseOpticalFlowVectorFieldAnalyzer non disponibile: %s", exc)

    try:
        from sef.builtin.analyzers.playback.TrackingPlaybackAnalyzer import TrackingPlaybackAnalyzer

        _try_register(
            registry,
            PluginCategory.ANALYZER,
            "tracking_playback",
            TrackingPlaybackAnalyzer,
            "Converte i campioni di tracking in un playback video-ready.",
        )
    except Exception as exc:
        log.warning("TrackingPlaybackAnalyzer non disponibile: %s", exc)

    try:
        from sef.builtin.analyzers.COCO_pose.COCOPoseStreamAnalyzer import COCOPoseStreamAnalyzer

        _try_register(
            registry,
            PluginCategory.ANALYZER,
            "coco_pose_stream",
            COCOPoseStreamAnalyzer,
            "Classifica movimento da skeleton COCO in streaming e mantiene i frame pose.",
        )
    except Exception as exc:
        log.warning("COCOPoseStreamAnalyzer non disponibile: %s", exc)

    try:
        from sef.builtin.analyzers.ArUco.ArucoMarkerDisplacementAnalyzer import ArucoMarkerDisplacementAnalyzer

        _try_register(
            registry,
            PluginCategory.ANALYZER,
            "aruco_displacement",
            ArucoMarkerDisplacementAnalyzer,
            "Calcola displacement 2D dei marker ArUco rispetto alla posizione iniziale.",
        )
    except Exception as exc:
        log.warning("ArucoMarkerDisplacementAnalyzer non disponibile: %s", exc)

    try:
        from sef.builtin.analyzers.ArUco.ArucoMarkerRelativeMotionAnalyzer import ArucoMarkerRelativeMotionAnalyzer

        _try_register(
            registry,
            PluginCategory.ANALYZER,
            "aruco_relative_motion",
            ArucoMarkerRelativeMotionAnalyzer,
            "Calcola la variazione di distanza tra coppie di marker ArUco.",
        )
    except Exception as exc:
        log.warning("ArucoMarkerRelativeMotionAnalyzer non disponibile: %s", exc)

    # ── Visualizers ───────────────────────────────────────────────────────────
    try:
        from sef.builtin.visualizers.Matplotlib.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer

        _try_register(registry, PluginCategory.VISUALIZER, "matplotlib_function", MatplotlibFunctionVisualizer, "Grafico linea/scatter Matplotlib.")
    except Exception as exc:
        log.warning("MatplotlibFunctionVisualizer non disponibile: %s", exc)

    try:
        from sef.builtin.visualizers.Matplotlib.MatplotlibHistogramVisualizer import MatplotlibHistogramVisualizer

        _try_register(
            registry, PluginCategory.VISUALIZER, "matplotlib_histogram", MatplotlibHistogramVisualizer, "Istogramma/bar chart Matplotlib."
        )
    except Exception as exc:
        log.warning("MatplotlibHistogramVisualizer non disponibile: %s", exc)

    try:
        from sef.builtin.visualizers.Matplotlib.MatplotlibTrajectoryVisualizer import MatplotlibTrajectoryVisualizer

        _try_register(
            registry, PluginCategory.VISUALIZER, "matplotlib_trajectory", MatplotlibTrajectoryVisualizer, "Visualizzazione traiettoria Matplotlib."
        )
    except Exception as exc:
        log.warning("MatplotlibTrajectoryVisualizer non disponibile: %s", exc)

    try:
        from sef.builtin.visualizers.Matplotlib.MatplotlibVectorFieldVisualizer import MatplotlibVectorFieldVisualizer

        _try_register(
            registry,
            PluginCategory.VISUALIZER,
            "matplotlib_vector_field",
            MatplotlibVectorFieldVisualizer,
            "Visualizzazione campi vettoriali Matplotlib.",
        )
    except Exception as exc:
        log.warning("MatplotlibVectorFieldVisualizer non disponibile: %s", exc)

    try:
        from sef.builtin.visualizers.TrackingVideoVisualizer import TrackingVideoVisualizer

        _try_register(
            registry,
            PluginCategory.VISUALIZER,
            "tracking_video",
            TrackingVideoVisualizer,
            "Ricostruisce un video annotato a partire dai campioni di tracking.",
        )
    except Exception as exc:
        log.warning("TrackingVideoVisualizer non disponibile: %s", exc)

    try:
        from sef.builtin.visualizers.ArUco.ArucoAnnotatedVideoVisualizer import ArucoAnnotatedVideoVisualizer

        _try_register(
            registry,
            PluginCategory.VISUALIZER,
            "aruco_annotated_video",
            ArucoAnnotatedVideoVisualizer,
            "Video annotato con corners, centri e id dei marker ArUco.",
        )
    except Exception as exc:
        log.warning("ArucoAnnotatedVideoVisualizer non disponibile: %s", exc)

    try:
        from sef.builtin.visualizers.COCO_pose.OpenCVCOCOPoseRealtimeVisualizer import OpenCVCOCOPoseRealtimeVisualizer

        _try_register(
            registry,
            PluginCategory.VISUALIZER,
            "opencv_coco_pose_realtime",
            OpenCVCOCOPoseRealtimeVisualizer,
            "Preview live OpenCV dei keypoint COCO.",
        )
    except Exception as exc:
        log.warning("OpenCVCOCOPoseRealtimeVisualizer non disponibile: %s", exc)

    try:
        from sef.builtin.visualizers.COCO_pose.OpenCVCOCOTennisPoseRealtimeVisualizer import OpenCVCOCOTennisPoseRealtimeVisualizer

        _try_register(
            registry,
            PluginCategory.VISUALIZER,
            "opencv_coco_tennis_pose_realtime",
            OpenCVCOCOTennisPoseRealtimeVisualizer,
            "Preview live OpenCV dei keypoint COCO con movimento tennis classificato.",
        )
    except Exception as exc:
        log.warning("OpenCVCOCOTennisPoseRealtimeVisualizer non disponibile: %s", exc)

    try:
        from sef.builtin.visualizers.COCO_pose.RealtimeCOCOPoseFrameVisualizer import RealtimeCOCOPoseFrameVisualizer
        from ui.services.realtime_preview_service import sink_for_id

        def streamlit_coco_pose_realtime_factory(config=None, sink_id=None):
            return RealtimeCOCOPoseFrameVisualizer(config=config, sink=sink_for_id(sink_id))

        _try_register(
            registry,
            PluginCategory.VISUALIZER,
            "streamlit_coco_pose_realtime",
            streamlit_coco_pose_realtime_factory,
            "Preview realtime browser-compatible dei keypoint COCO per Streamlit.",
        )
    except Exception as exc:
        log.warning("RealtimeCOCOPoseFrameVisualizer non disponibile: %s", exc)

    try:
        from sef.builtin.visualizers.Matplotlib.MatplotlibArucoMotionVisualizer import MatplotlibArucoMotionVisualizer

        _try_register(
            registry,
            PluginCategory.VISUALIZER,
            "aruco_motion_plot",
            MatplotlibArucoMotionVisualizer,
            "Plot del displacement e del moto relativo dei marker ArUco.",
        )
    except Exception as exc:
        log.warning("MatplotlibArucoMotionVisualizer non disponibile: %s", exc)

    # ── Intermediate Frame visualizers ────────────────────────────────────────
    try:
        from sef.builtin.visualizers.intermediate_frames.IntermediateFramesVisualizer import IntermediateFramesVisualizer

        _try_register(
            registry,
            PluginCategory.VISUALIZER,
            "intermediate_frames",
            IntermediateFramesVisualizer,
            "Render each captured preprocessing snapshot as a comparison PNG.",
        )
    except Exception as exc:
        log.warning("IntermediateFramesVisualizer non disponibile: %s", exc)

    try:
        from sef.builtin.visualizers.intermediate_frames.IntermediateFramesGridVisualizer import IntermediateFramesGridVisualizer

        _try_register(
            registry,
            PluginCategory.VISUALIZER,
            "intermediate_frames_grid",
            IntermediateFramesGridVisualizer,
            "Render captured preprocessing snapshots as a bounded comparison grid.",
        )
    except Exception as exc:
        log.warning("IntermediateFramesGridVisualizer non disponibile: %s", exc)

    # ── Branching rules ───────────────────────────────────────────────────────
    try:
        from sef.builtin.branching_rules.NewTrackBranchingRule import NewTrackBranchingRule

        _try_register(
            registry,
            PluginCategory.BRANCHING_RULE,
            "default_track_branch",
            NewTrackBranchingRule,
            "Branch once when the primary multi-object tracker creates its seed track.",
        )
    except Exception as exc:
        log.warning("NewTrackBranchingRule non disponibile: %s", exc)

    try:
        from sef.builtin.visualizers.Matplotlib.MatplotlibHeatmapVisualizer import MatplotlibHeatmapVisualizer

        _try_register(
            registry,
            PluginCategory.VISUALIZER,
            "matplotlib_heatmap",
            MatplotlibHeatmapVisualizer,
            "Visualizzazione heatmap Matplotlib.",
        )
    except Exception as exc:
        log.warning("MatplotlibHeatmapVisualizer non disponibile: %s", exc)

    loaded = len(registry.list())
    log.info("PluginRegistry pronto: %d plugin caricati.", loaded)
    return registry
