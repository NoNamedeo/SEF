"""High-level Pythonic SEF API."""

from sef.api.config import load_config, normalize_config
from sef.api.decorators import analyzer, cleaner, frame_extractor, processor, signal_extractor, visualizer
from sef.api.pipeline import PipelineFacade, from_config, pipeline, video, webcam
from sef.api.registry import default_registry, register_user_plugin

__all__ = [
    "PipelineFacade",
    "analyzer",
    "cleaner",
    "default_registry",
    "frame_extractor",
    "from_config",
    "load_config",
    "normalize_config",
    "pipeline",
    "processor",
    "register_user_plugin",
    "signal_extractor",
    "video",
    "visualizer",
    "webcam",
]
