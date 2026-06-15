from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.artifacts.Frame import Frame
from sef.core.interfaces.IAnalyzer import IAnalyzer
from sef.core.interfaces.IData import IData
from sef.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from sef.core.interfaces.IFrameExtractor import IFrameExtractor
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalCleaner import ISignalCleaner
from sef.core.interfaces.ISignalExtractor import ISignalExtractor
from sef.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.interfaces.IVisualizer import IVisualizer
from sef.core.visualization.VisualArtifact import VisualArtifact
from sef.core.visualization.VisualizationContext import VisualizationContext


def resolve_function_capabilities(
    function: Callable[..., Any],
    explicit: StageCapabilities | None,
    fallback: StageCapabilities | None,
) -> StageCapabilities | None:
    """Resolve explicit, function-level, or contract-default capabilities."""
    if explicit is not None:
        return explicit
    declared = getattr(function, "capabilities", None)
    if isinstance(declared, StageCapabilities):
        return declared
    return fallback


class FunctionFrameExtractor(IFrameExtractor):
    """Adapt a plain callable into an ``IFrameExtractor`` plugin."""

    def __init__(
        self,
        function: Callable[..., FrameBuffer],
        params: dict[str, Any],
        *,
        capabilities: StageCapabilities | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)
        resolved = resolve_function_capabilities(function, capabilities, type(self).capabilities)
        if resolved is not None:
            self.capabilities = resolved

    def extract(self) -> FrameBuffer:
        return self._function(**self._params)


class FunctionSignalExtractor(ISignalExtractor):
    """Adapt a plain callable into an ``ISignalExtractor`` plugin."""

    def __init__(
        self,
        function: Callable[..., ISignal],
        params: dict[str, Any],
        *,
        capabilities: StageCapabilities | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)
        resolved = resolve_function_capabilities(function, capabilities, type(self).capabilities)
        if resolved is not None:
            self.capabilities = resolved

    def extract(self, buffer: FrameBuffer) -> ISignal:
        return self._function(buffer, **self._params)


class FunctionSignalCleaner(ISignalCleaner):
    """Adapt a plain callable into an ``ISignalCleaner`` plugin."""

    def __init__(
        self,
        function: Callable[..., ISignal],
        params: dict[str, Any],
        *,
        capabilities: StageCapabilities | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)
        resolved = resolve_function_capabilities(function, capabilities, type(self).capabilities)
        if resolved is not None:
            self.capabilities = resolved

    def clean(self, signal: ISignal) -> ISignal:
        return self._function(signal, **self._params)


class FunctionAnalyzer(IAnalyzer):
    """Adapt a plain callable into an ``IAnalyzer`` plugin."""

    def __init__(
        self,
        function: Callable[..., IData],
        params: dict[str, Any],
        *,
        capabilities: StageCapabilities | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)
        resolved = resolve_function_capabilities(function, capabilities, type(self).capabilities)
        if resolved is not None:
            self.capabilities = resolved

    def analyze(self, signal: ISignal) -> IData:
        return self._function(signal, **self._params)


class FunctionVisualizer(IVisualizer):
    """Adapt a plain callable into an ``IVisualizer`` plugin."""

    def __init__(
        self,
        function: Callable[..., VisualArtifact | tuple[VisualArtifact, ...] | list[VisualArtifact]],
        params: dict[str, Any],
        *,
        capabilities: StageCapabilities | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)
        self._accepts_context = "context" in inspect.signature(function).parameters
        resolved = resolve_function_capabilities(function, capabilities, type(self).capabilities)
        if resolved is not None:
            self.capabilities = resolved

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
        capabilities: StageCapabilities | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)
        self._accepts_frame = accepts_frame
        resolved = resolve_function_capabilities(function, capabilities, getattr(type(self), "capabilities", None))
        if resolved is not None:
            self.capabilities = resolved

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


class FunctionFrameBufferProcessor(IFrameBufferProcessor):
    """Adapt a plain callable into an ``IFrameBufferProcessor`` plugin."""

    def __init__(
        self,
        function: Callable[..., FrameBuffer],
        params: dict[str, Any],
        *,
        capabilities: StageCapabilities | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._function = function
        self._params = dict(params)
        resolved = resolve_function_capabilities(function, capabilities, type(self).capabilities)
        if resolved is not None:
            self.capabilities = resolved

    def process(self, buffer: FrameBuffer) -> FrameBuffer:
        return self._function(buffer, **self._params)
