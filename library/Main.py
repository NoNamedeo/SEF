"""
main.py — Esempi di utilizzo del framework Pipeline

Unico entry-point pubblico: PipelineOrchestrator.
Pipeline è un dettaglio implementativo interno — non viene mai esposta.

Mostra due modalità di costruzione:
  1. FluentPipelineBuilder  → costruzione programmatica (test, script)
  2. ConfigPipelineBuilder  → costruzione da dizionario / YAML (produzione, UI)

In entrambi i casi .build() restituisce un PipelineOrchestrator.
"""

import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


# ── Import framework ─────────────────────────────────────────────────────────
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator, PipelineEvent
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.plugins.PluginRegistry import create_builtin_registry

# Componenti concreti — usati solo nel fluent builder
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.frame_cleaners.OpenCVGrayFrameCleaner import OpenCVGrayFrameCleaner
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
from library.signal_cleaners.OpenCVMovingAverageCleaner import OpenCVMovingAverageCleaner
from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer


VIDEO_PATH = "videos/Baloons.mp4"
ROI        = (100, 200, 50, 80)   # x, y, w, h


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
        .build(max_retries=1) # -> PipelineOrchestrator
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
        ConfigPipelineBuilder(create_builtin_registry())
        .build(config)                    # -> PipelineOrchestrator
    )

    # Aggiunge eventi prima di run()
    orchestrator.subscribe(PipelineEvent.AFTER_RUN,
                           lambda p: log.info("AFTER_RUN: %d risultati", len(p.results)))
    orchestrator.subscribe(PipelineEvent.ON_ERROR,
                           lambda p: log.error("ON_ERROR: %s", p.error))

    results = orchestrator.run()
    log.info("Risultati: %s", results)

    # Pipeline secondaria condizionale:
    # se i risultati primari soddisfano una condizione, lancia un'analisi
    # piu' dettagliata (es. KeypointAnalyzer per la biomeccanica).
    if _needs_secondary_analysis(results):
        log.info("Avvio pipeline secondaria (analisi dettagliata).")
        secondary_context = PipelineContext(
            frame_extractor  = OpenCVBufferedFrameExtractor(VIDEO_PATH),
            signal_extractor = OpenCVBufferedSignalExtractor(start_box=ROI),
            analyzers        = [VerticalPositionAnalyzer()],
            # aggiungere qui KeypointAnalyzer() una volta registrato come plugin
        )
        secondary_results = orchestrator.run_secondary(secondary_context)
        log.info("Risultati secondari: %s", secondary_results)


def _needs_secondary_analysis(results) -> bool:
    return len(results) > 0


# ════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # example_fluent()
    example_config()