from abc import ABC, abstractmethod
from typing import Any

from library.core.artifacts.Frame import Frame


class ISingleFrameProcessor(ABC):
    """
    Contract for stateless or local frame transformations.

    Single-frame processors are adapted into the frame-buffer pipeline by the
    core runtime. Use this contract when processing one frame does not require
    future or past frames.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Store plugin-specific processor configuration."""
        self.config = config or {}

    @abstractmethod
    def process(self, frame: Frame) -> Frame:
        """
        Transform one frame.

        Parameters
        ----------
        frame:
            Input frame.

        Returns
        -------
        Frame
            Processed frame. Implementations may preserve metadata or add
            metadata needed by later stages.
        """
