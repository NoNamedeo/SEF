"""
Main.py — Esempi di utilizzo del framework Pipeline

Unico entry-point pubblico: PipelineOrchestrator.
Pipeline è un dettaglio implementativo interno — non viene mai esposta.

Tre esempi progressivamente più ricchi:

  1. FluentPipelineBuilder        → costruzione programmatica (script, test)
  2. ConfigPipelineBuilder        → costruzione dichiarativa da YAML
  3. Pipeline parallele su eventi → IEventEmitter + IBranchingRule + ThreadPoolExecutor
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import yaml

from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.SignalSample import BoundingBox, SignalSample
from library.core.events.DomainEvent import DomainEvent
from library.core.interfaces.IEventEmitter import IEventEmitter
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.pipeline.IBranchingRule import IBranchingRule
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineEvent, PipelineOrchestrator
from library.core.plugins.PluginRegistry import create_builtin_registry
from library.frame_cleaners.OpenCVGrayFrameCleaner import OpenCVGrayFrameCleaner
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_cleaners.OpenCVMovingAverageCleaner import OpenCVMovingAverageCleaner
from library.signal_extractors.OpenCVBufferedSignalExtractor import (
    OpenCVBufferedSignalExtractor,
)
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


VIDEO_PATH = "videos/Baloons.mp4"
ROI = (100, 200, 50, 80)  # x, y, w, h


# ════════════════════════════════════════════════════════════════════════════
# 1. FLUENT BUILDER
#    Costruzione programmatica — ideale per script, test, notebook.
#    .build() → PipelineOrchestrator  (Pipeline è interna, invisibile)
# ════════════════════════════════════════════════════════════════════════════


def example_fluent() -> None:
    log.info("=== Esempio 1: FluentPipelineBuilder ===")

    orchestrator: PipelineOrchestrator = (
        FluentPipelineBuilder()
        .with_frame_extractor(OpenCVBufferedFrameExtractor(VIDEO_PATH))
        .add_frame_cleaner(OpenCVGrayFrameCleaner())
        .with_signal_extractor(OpenCVBufferedSignalExtractor(start_box=ROI))
        .add_signal_cleaner(OpenCVMovingAverageCleaner(window_size=5))
        .add_analyzer(VerticalPositionAnalyzer())
        .add_visualizer(MatplotlibFunctionVisualizer())
        .with_max_retries(1)
        .build()  # → PipelineOrchestrator
    )

    results = orchestrator.run()
    log.info("Risultati: %s", results)


# ════════════════════════════════════════════════════════════════════════════
# 2. CONFIG BUILDER
#    Costruzione dichiarativa da dizionario — in produzione arriva da YAML.
#    Nessun import concreto: tutto risolto dal PluginRegistry per nome.
#    .build(config) → PipelineOrchestrator
# ════════════════════════════════════════════════════════════════════════════


def example_config() -> None:
    log.info("=== Esempio 2: ConfigPipelineBuilder ===")

    config = yaml.safe_load(open("pipeline.yaml"))

    orchestrator: PipelineOrchestrator = (
        ConfigPipelineBuilder(create_builtin_registry()).build(config)  # → PipelineOrchestrator
    )

    # Lifecycle hooks: per logging, metriche, UI update ecc.
    orchestrator.subscribe(
        PipelineEvent.AFTER_RUN,
        lambda p: log.info("AFTER_RUN: %d risultati", len(p.results)),
    )
    orchestrator.subscribe(
        PipelineEvent.ON_ERROR,
        lambda p: log.error("ON_ERROR: %s", p.error),
    )

    results = orchestrator.run()
    log.info("Risultati: %s", results)


# ════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE PARALLELE SU EVENTI
#
#    Scenario realistico (video palloncini):
#    ┌─────────────────────────────────────────────────────────────────┐
#    │  Pipeline PRIMARIA                                              │
#    │  Video → GrayFrames → Tracker CSRT → MovingAvg → VerticalPos  │
#    │                │                                                │
#    │                └─ se tracking perso (box nulla) ──────────────┐│
#    │                └─ se il palloncino sale troppo in alto ────────┤│
#    │                                                                ││
#    │  Pipeline SECONDARIA A  (spawned su ThreadPoolExecutor)        ││
#    │  stesso video, ROI allargato, analisi posizione verticale      ││
#    │                                                                ││
#    │  Pipeline SECONDARIA B  (spawned su ThreadPoolExecutor)        ││
#    │  stesso video, ROI più grande, analisi posizione verticale     ││
#    └─────────────────────────────────────────────────────────────────┘
#
#    Due componenti nuovi:
#      • EventAwareTracker         — ISignalExtractor + IEventEmitter
#      • TrackingLostBranch        — IBranchingRule (Strategy)
#      • BalloonAscendingBranch    — IBranchingRule (Strategy)
# ════════════════════════════════════════════════════════════════════════════


# ── Componente personalizzato: SignalExtractor che emette eventi ─────────────


class EventAwareTracker(ISignalExtractor, IEventEmitter):
    """
    Tracker OpenCV con emissione di eventi di dominio.

    Estende ISignalExtractor (contratto pipeline) e
    IEventEmitter (mixin — aggiunge self.emit() senza modificare l'interfaccia).

    Eventi emessi:
      "tracking_lost"      → quando tracker.update() restituisce success=False
      "balloon_ascending"  → quando il centroide supera la soglia verticale
    """

    TRACKING_LOST_EVENT = "tracking_lost"
    BALLOON_ASCENDING_EVENT = "balloon_ascending"

    def __init__(
        self,
        start_box: BoundingBox,
        tracker_factory: Callable[[], Any] | None = None,
        ascending_threshold_y: float = 150.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        # Chiama ENTRAMBI gli __init__ della catena MRO
        ISignalExtractor.__init__(self, config)
        self._start_box = start_box
        self._tracker_factory = tracker_factory
        self._ascending_threshold_y = ascending_threshold_y

    # ── ISignalExtractor ─────────────────────────────────────────────────────

    def extract(self, buffer: FrameBuffer) -> ISignal:
        """
        Traccia il palloncino frame per frame.

        Emette eventi durante l'estrazione:
          • "tracking_lost"      se il tracker perde il target
          • "balloon_ascending"  se il centroide Y scende sotto la soglia
            (in coordinate immagine Y=0 è in alto, quindi valori bassi = alto)
        """
        tracker = self._build_tracker()
        samples: list[SignalSample] = []
        current_box: BoundingBox | None = None

        for position, frame in enumerate(buffer):
            frame_index = frame.index if frame.index is not None else position

            if position == 0:
                tracker.init(frame.frame, self._start_box)
                current_box = self._start_box
            else:
                success, updated_box = tracker.update(frame.frame)

                if success:
                    x, y, w, h = (int(v) for v in updated_box)
                    current_box = (x, y, w, h)
                else:
                    current_box = None
                    # ── Evento di dominio: tracking perso ────────────────────
                    self.emit(
                        self.TRACKING_LOST_EVENT,
                        {
                            "frame_index": frame_index,
                            "last_box": current_box,
                            "reason": "tracker.update() returned success=False",
                        },
                    )
                    log.warning("EventAwareTracker: tracking perso al frame %d", frame_index)

            centroid = None
            if current_box is not None:
                bx, by, bw, bh = current_box
                centroid = (bx + bw / 2.0, by + bh / 2.0)

                # ── Evento di dominio: palloncino in salita ──────────────────
                if centroid[1] < self._ascending_threshold_y:
                    self.emit(
                        self.BALLOON_ASCENDING_EVENT,
                        {
                            "frame_index": frame_index,
                            "centroid_y": centroid[1],
                            "threshold_y": self._ascending_threshold_y,
                        },
                    )

            samples.append(
                SignalSample(
                    frame_index=frame_index,
                    box=current_box,
                    centroid=centroid,
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata=dict(frame.metadata),
                )
            )

        return Signal(samples)

    def _build_tracker(self) -> Any:
        if self._tracker_factory is not None:
            return self._tracker_factory()
        # Fallback: usa il meccanismo interno di OpenCVBufferedSignalExtractor
        return OpenCVBufferedSignalExtractor._create_tracker("CSRT")


# ── Branching rules ──────────────────────────────────────────────────────────


class TrackingLostBranch(IBranchingRule):
    """
    Quando il tracker perde il target, spawna UNA SOLA pipeline secondaria
    che ritenta l'analisi con una ROI allargata e un tracker diverso.

    Il guard ``_triggered`` è fondamentale: senza di esso ogni frame di
    tracking perso emetterebbe un evento e spawnerebbe una pipeline a sé,
    saturando il ThreadPoolExecutor e le risorse di sistema.

    Pattern Strategy: incapsula la decisione di branching
    senza modificare l'Orchestrator.
    """

    def __init__(self, video_path: str, wider_roi: BoundingBox) -> None:
        self._video_path = video_path
        self._wider_roi = wider_roi
        self._triggered = False  # guardia: un solo spawn per sessione

    def matches(self, event: DomainEvent) -> bool:
        if event.event_type != EventAwareTracker.TRACKING_LOST_EVENT:
            return False
        if self._triggered:
            return False  # tracking perso di nuovo — non rispawnare
        self._triggered = True
        return True

    def build_context(self, event: DomainEvent) -> PipelineContext:
        lost_frame = event.payload.get("frame_index", 0)
        log.info(
            "TrackingLostBranch: avvio pipeline di recupero (frame %d, ROI allargata %s)",
            lost_frame,
            self._wider_roi,
        )
        return PipelineContext(
            frame_extractor=OpenCVBufferedFrameExtractor(
                self._video_path,
                config={"max_frames": None},
            ),
            signal_extractor=OpenCVBufferedSignalExtractor(
                tracker_type="KCF",  # tracker diverso come fallback
                start_box=self._wider_roi,
            ),
            signal_cleaners=[OpenCVMovingAverageCleaner(window_size=7)],
            analyzers=[VerticalPositionAnalyzer()],
            visualizers=[MatplotlibFunctionVisualizer()],
        )


class BalloonAscendingBranch(IBranchingRule):
    """
    Quando il palloncino supera la soglia verticale, spawna una pipeline
    secondaria di analisi fine con finestra temporale ristretta.

    Viene attivata UNA SOLA VOLTA: usa un flag interno per evitare
    di spawnare una pipeline per ogni frame in salita.
    """

    def __init__(self, video_path: str, roi: BoundingBox) -> None:
        self._video_path = video_path
        self._roi = roi
        self._triggered = False  # guardia: un solo spawn per sessione

    def matches(self, event: DomainEvent) -> bool:
        if event.event_type != EventAwareTracker.BALLOON_ASCENDING_EVENT:
            return False
        if self._triggered:
            return False  # non spawnare di nuovo
        self._triggered = True
        return True

    def build_context(self, event: DomainEvent) -> PipelineContext:
        centroid_y = event.payload.get("centroid_y", 0.0)
        log.info(
            "BalloonAscendingBranch: palloncino in salita (y=%.1f) — avvio analisi dettagliata",
            centroid_y,
        )
        return PipelineContext(
            frame_extractor=OpenCVBufferedFrameExtractor(
                self._video_path,
                config={"stride": 1},
            ),
            frame_cleaners=[OpenCVGrayFrameCleaner()],
            signal_extractor=OpenCVBufferedSignalExtractor(
                tracker_type="CSRT",
                start_box=self._roi,
            ),
            signal_cleaners=[OpenCVMovingAverageCleaner(window_size=3)],
            analyzers=[VerticalPositionAnalyzer(config={"use_timestamps": True})],
        )


# ── Esempio 3 completo ───────────────────────────────────────────────────────


def example_parallel_pipelines() -> None:
    """
    Esempio completo: pipeline primaria + pipeline secondarie parallele.

    Flusso:
      1. FluentPipelineBuilder costruisce l'Orchestrator con:
           - EventAwareTracker    → emette eventi di dominio durante extract()
           - TrackingLostBranch   → spawna pipeline di recupero su "tracking_lost"
           - BalloonAscendingBranch → spawna analisi fine su "balloon_ascending"
      2. orchestrator.run() → esegue la pipeline primaria.
           • Internamente Pipeline inietta l'EventBus nell'EventAwareTracker.
           • Ogni evento arriva all'Orchestrator → valuta le BranchingRules.
           • Le pipeline secondarie partono su un ThreadPoolExecutor in parallelo.
      3. orchestrator.collect_secondary_results() → attende e raccoglie.
      4. orchestrator.shutdown() → rilascia il ThreadPoolExecutor.
    """
    log.info("=== Esempio 3: Pipeline parallele su eventi ===")

    # ROI allargata per il recupero del tracking
    WIDER_ROI = (
        max(0, ROI[0] - 30),
        max(0, ROI[1] - 30),
        ROI[2] + 60,
        ROI[3] + 60,
    )

    # ── Costruzione orchestrator ─────────────────────────────────────────────
    orchestrator: PipelineOrchestrator = (
        FluentPipelineBuilder()
        # ── Componenti pipeline primaria ─────────────────────────────────
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(VIDEO_PATH),
        )
        .add_frame_cleaner(OpenCVGrayFrameCleaner())
        .with_signal_extractor(
            # ISignalExtractor + IEventEmitter: emette eventi durante extract()
            EventAwareTracker(
                start_box=ROI,
                ascending_threshold_y=150.0,
            )
        )
        .add_signal_cleaner(OpenCVMovingAverageCleaner(window_size=5))
        .add_analyzer(VerticalPositionAnalyzer())
        .add_visualizer(MatplotlibFunctionVisualizer())
        # ── Branching rules (Strategy Pattern) ──────────────────────────
        # Ogni regola è valutata per OGNI evento emesso dai componenti.
        # Se matches() → True, build_context() viene chiamato e la pipeline
        # secondaria è submessa al ThreadPoolExecutor.
        .add_branching_rule(TrackingLostBranch(VIDEO_PATH, WIDER_ROI))
        .add_branching_rule(BalloonAscendingBranch(VIDEO_PATH, ROI))
        # ── Retry policy per la pipeline primaria ────────────────────────
        .with_max_retries(1)  # → FixedRetryPolicy(1) internamente
        .build()
    )

    # ── Lifecycle hooks (log, metriche, UI) ──────────────────────────────────
    orchestrator.subscribe(
        PipelineEvent.BEFORE_RUN,
        lambda p: log.info("▶ Pipeline primaria avviata"),
    )
    orchestrator.subscribe(
        PipelineEvent.AFTER_RUN,
        lambda p: log.info(
            "✓ Pipeline primaria completata — %d risultati",
            len(p.results),
        ),
    )
    orchestrator.subscribe(
        PipelineEvent.ON_ERROR,
        lambda p: log.error(
            "✗ Errore alla pipeline primaria (attempt %d): %s",
            p.attempt,
            p.error,
        ),
    )
    orchestrator.subscribe(
        PipelineEvent.ON_RETRY,
        lambda p: log.warning("↻ Retry n.%d della pipeline primaria", p.attempt),
    )

    # ── 1. Esecuzione pipeline PRIMARIA ──────────────────────────────────────
    #    Durante run(), l'EventBus è iniettato nell'EventAwareTracker.
    #    Gli eventi emessi durante extract() vengono catturati in real-time
    #    dall'Orchestrator che valuta le BranchingRules e spawna le secondarie.
    primary_results = orchestrator.run()

    log.info("─" * 60)
    log.info("Risultati pipeline PRIMARIA (%d analyzer):", len(primary_results))
    for i, data in enumerate(primary_results):
        log.info("  [%d] %s — %d punti", i, data.title, len(data.x))

    # ── 2. Raccolta risultati pipeline SECONDARIE ─────────────────────────────
    #    collect_secondary_results() attende il completamento di tutte le
    #    pipeline spawnate automaticamente dalle BranchingRules.
    #    timeout=30 → aspetta al massimo 30 secondi per ciascuna.
    log.info("─" * 60)
    log.info(
        "Pipeline secondarie in volo: %d — attendo completamento...",
        orchestrator.pending_secondary_count,
    )

    secondary_results = orchestrator.collect_secondary_results(timeout=30)

    log.info("Risultati pipeline SECONDARIE (%d pipeline):", len(secondary_results))
    for i, results in enumerate(secondary_results):
        for j, data in enumerate(results):
            log.info(
                "  Pipeline secondaria [%d] → analyzer [%d]: %s — %d punti",
                i,
                j,
                data.title,
                len(data.x),
            )

    # ── 3. Shutdown ThreadPoolExecutor ────────────────────────────────────────
    orchestrator.shutdown(wait=True)
    log.info("Orchestrator spento correttamente.")


# ════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    example_fluent()
    # example_config()
    # example_parallel_pipelines()
