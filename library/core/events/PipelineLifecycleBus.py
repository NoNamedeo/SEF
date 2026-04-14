from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Callable

from library.core.interfaces.IData import IData


class PipelineLifecycleEvent(StrEnum):
    BEFORE_RUN = auto()
    AFTER_RUN = auto()
    ON_ERROR = auto()
    ON_RETRY = auto()


@dataclass
class PipelineLifecyclePayload:
    event: PipelineLifecycleEvent
    pipeline_id: str
    results: list[IData] = field(default_factory=list)
    error: Exception | None = field(default=None)
    attempt: int = field(default=1)

    @property
    def event_type(self) -> str:
        return str(self.event)


LifecycleEventHandler = Callable[[PipelineLifecyclePayload], None]
