from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.abstractions.IBranchingRule import IBranchingRule
from library.core.abstractions.IEventEmitter import IEventEmitter
from library.core.events.DomainEvent import DomainEvent
from library.core.events.EventBus import EventBus
from library.core.events.PipelineLifecycleBus import PipelineLifecycleBus
from library.core.pipeline.BranchingCoordinator import BranchingCoordinator
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.PipelineOrchestrator import PipelineEvent, PipelineOrchestrator
from library.core.plugins.PluginRegistry import PluginRegistry, create_builtin_registry
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_cleaners.OpenCVMovingAverageCleaner import OpenCVMovingAverageCleaner
from library.signal_extractors.OpenCVBufferedSignalExtractor import (
    OpenCVBufferedSignalExtractor,
)
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer

__all__ = [
    # ── Builders ─────────────────────────────────────────────────────────
    "FluentPipelineBuilder",
    "ConfigPipelineBuilder",
    # ── Orchestrator & coordination ──────────────────────────────────────
    "PipelineOrchestrator",
    "PipelineEvent",
    "PipelineLifecycleBus",
    "BranchingCoordinator",
    # ── Domain events ────────────────────────────────────────────────────
    "EventBus",
    "DomainEvent",
    # ── Abstractions (user-facing) ───────────────────────────────────────
    "IEventEmitter",
    "IBranchingRule",
    # ── Plugin system ────────────────────────────────────────────────────
    "PluginRegistry",
    "create_builtin_registry",
    # ── Concrete components ──────────────────────────────────────────────
    "VerticalPositionAnalyzer",
    "OpenCVBufferedFrameExtractor",
    "OpenCVBufferedSignalExtractor",
    "OpenCVMovingAverageCleaner",
    "MatplotlibFunctionVisualizer",
]
