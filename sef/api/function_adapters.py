from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from library.core.artifacts.Frame import Frame
from library.core.artifacts.buffer.FrameBuffer import FrameBuffer
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.visualization.VisualArtifact import VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext


class FunctionFrameExtractor(IFrameExtractor):
    """Adapt a plain callable into an ``IFrameExtractor`` plugin."""

    def __init__(self, function: Callable[..., FrameBuffer], params: dict[str, Any], config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)

    def extract(self) -> FrameBuffer:
        return self._function(**self._params)


class FunctionSignalExtractor(ISignalExtractor):
    """Adapt a plain callable into an ``ISignalExtractor`` plugin."""

    def __init__(self, function: Callable[..., ISignal], params: dict[str, Any], config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)

    def extract(self, buffer: FrameBuffer) -> ISignal:
        return self._function(buffer, **self._params)


class FunctionSignalCleaner(ISignalCleaner):
    """Adapt a plain callable into an ``ISignalCleaner`` plugin."""

    def __init__(self, function: Callable[..., ISignal], params: dict[str, Any], config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)

    def clean(self, signal: ISignal) -> ISignal:
        return self._function(signal, **self._params)


class FunctionAnalyzer(IAnalyzer):
    """Adapt a plain callable into an ``IAnalyzer`` plugin."""

    def __init__(self, function: Callable[..., IData], params: dict[str, Any], config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)

    def analyze(self, signal: ISignal) -> IData:
        return self._function(signal, **self._params)


class FunctionVisualizer(IVisualizer):
    """Adapt a plain callable into an ``IVisualizer`` plugin."""

    def __init__(
        self,
        function: Callable[..., VisualArtifact | tuple[VisualArtifact, ...] | list[VisualArtifact]],
        params: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)
        self._accepts_context = "context" in inspect.signature(function).parameters

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        if self._accepts_context:
            result = self._function(data, context=context, **self._params)
        else:
            result = self._function(data, **self._params)
        if isinstance(result, VisualArtifact):
            return (result,)
        return tuple(result)


class FunctionFrameProcessor(ISingleFrameProcessor):
    """
    Adapt a plain callable into a single-frame processor.

    By default the callable receives the frame image and may return either a
    new image array or a complete ``Frame``. Set ``accepts_frame=True`` when
    the callable needs access to indexes, timestamps, or metadata.
    """

    def __init__(
        self,
        function: Callable[..., Any],
        params: dict[str, Any],
        *,
        accepts_frame: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)
        self._accepts_frame = accepts_frame

    def process(self, frame: Frame) -> Frame:
        source = frame if self._accepts_frame else frame.image
        result = self._function(source, **self._params)
        if isinstance(result, Frame):
            return result
        return Frame(
            image=result,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata=dict(frame.metadata),
        )
