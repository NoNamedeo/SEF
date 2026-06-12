from __future__ import annotations

from abc import ABC, abstractmethod

from library.core.events.Event import Event
from library.core.pipeline.PipelineContext import PipelineContext


class IBranchingRule(ABC):
    """
    Strategy that decides IF and HOW to create a secondary pipeline
    in response to a domain event.

    Design rationale
    ----------------
    The Orchestrator holds a list of IBranchingRules.  When a domain
    event arrives on the EventBus, **each** rule is evaluated in order:

    1. ``matches(event)`` — should this event trigger a secondary pipeline?
    2. ``build_context(event)`` — build the PipelineContext for the new pipeline.

    This is the **Strategy pattern**: users define their own branching
    logic by subclassing IBranchingRule, without touching the Orchestrator.

    Example
    -------
    >>> class TrackingLostBranch(IBranchingRule):
    ...     def matches(self, event: Event) -> bool:
    ...         return event.event_type == "tracking_lost"
    ...
    ...     def build_context(self, event: Event) -> PipelineContext:
    ...         return PipelineContext(
    ...             frame_extractor=...,
    ...             signal_extractor=...,
    ...             analyzers=[...],
    ...         )
    """

    @abstractmethod
    def matches(self, event: Event) -> bool:
        """
        Return ``True`` if *event* should trigger a secondary pipeline.

        Parameters
        ----------
        event:
            The domain event published on the EventBus.

        Returns
        -------
        bool
            ``True``  -> ``build_context()`` will be called next.
            ``False`` -> this rule is skipped for this event.
        """

    @abstractmethod
    def build_context(self, event: Event) -> PipelineContext:
        """
        Build the PipelineContext for the secondary pipeline.

        Called only when ``matches(event)`` returned ``True``.

        Parameters
        ----------
        event:
            The same domain event that matched.

        Returns
        -------
        PipelineContext
            A fully configured, immutable context ready for execution
            by a secondary Pipeline.
        """
