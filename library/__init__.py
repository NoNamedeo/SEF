from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.interfaces.IEventEmitter import IEventEmitter
from library.core.interfaces.pipeline.IBranchingRule import IBranchingRule
from library.core.artifacts.PipelineEvent import PipelineEvent
from library.core.events.DomainEvent import DomainEvent
from library.core.events.EventBus import EventBus
from library.core.events.PipelineLifecycleBus import (
    PipelineLifecycleEvent,
    PipelineLifecyclePayload,
)
from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.interfaces.pipeline.IPipelineBuilder import IPipelineBuilder
from library.core.interfaces.pipeline.IPipelineMonitor import IPipelineMonitor
from library.core.interfaces.pipeline.IPipelineRunner import IPipelineRunner
from library.core.pipeline.BranchingCoordinator import BranchingCoordinator
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.DefaultPipelineBuilder import DefaultPipelineBuilder
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.InMemoryPipelineMonitor import InMemoryPipelineMonitor
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.ThreadedPipelineRunner import ThreadedPipelineRunner
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
    "BranchingCoordinator",
    # ── Concrete infrastructure ──────────────────────────────────────────
    "DefaultPipelineBuilder",
    "ThreadedPipelineRunner",
    "InMemoryPipelineMonitor",
    # ── Interfaces ───────────────────────────────────────────────────────
    "IEventBus",
    "IPipelineBuilder",
    "IPipelineRunner",
    "IPipelineMonitor",
    # ── Event system ────────────────────────────────────────────────────
    "EventBus",
    "DomainEvent",
    "PipelineEvent",
    "PipelineLifecycleEvent",
    "PipelineLifecyclePayload",
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
