from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StageCapabilities:
    """
    Stable execution contract declared by every pipeline stage.

    The pipeline runtime uses these flags to build an execution plan without
    relying on implementation details such as ad-hoc attributes. Concrete
    stages should be conservative: declare streaming only when they can produce
    output progressively without requiring the full upstream sequence.
    """

    supports_streaming: bool
    requires_complete_sequence: bool
    stateful: bool = False
    preserves_order: bool = True
    supports_frame_parallelism: bool = False
    realtime_safe: bool = False

    @classmethod
    def batch(
        cls,
        *,
        stateful: bool = True,
        preserves_order: bool = True,
        supports_frame_parallelism: bool = False,
        realtime_safe: bool = False,
    ) -> StageCapabilities:
        """Return capabilities for stages that require a complete input sequence."""
        return cls(
            supports_streaming=False,
            requires_complete_sequence=True,
            stateful=stateful,
            preserves_order=preserves_order,
            supports_frame_parallelism=supports_frame_parallelism,
            realtime_safe=realtime_safe,
        )

    @classmethod
    def streaming(
        cls,
        *,
        stateful: bool = False,
        preserves_order: bool = True,
        supports_frame_parallelism: bool = False,
        realtime_safe: bool = True,
    ) -> StageCapabilities:
        """Return capabilities for stages that can process input progressively."""
        return cls(
            supports_streaming=True,
            requires_complete_sequence=False,
            stateful=stateful,
            preserves_order=preserves_order,
            supports_frame_parallelism=supports_frame_parallelism,
            realtime_safe=realtime_safe,
        )

    def as_dict(self) -> dict[str, bool]:
        """Return a JSON-safe representation used by execution plans and UI."""
        return {
            "supports_streaming": self.supports_streaming,
            "requires_complete_sequence": self.requires_complete_sequence,
            "stateful": self.stateful,
            "preserves_order": self.preserves_order,
            "supports_frame_parallelism": self.supports_frame_parallelism,
            "realtime_safe": self.realtime_safe,
        }
