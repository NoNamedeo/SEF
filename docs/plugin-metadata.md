# Plugin Metadata

Plugin metadata is a JSON-friendly convention used by the CLI, documentation,
and future UI catalogs. The registry stores metadata as an immutable mapping but
does not enforce a schema. This keeps plugin authoring flexible while giving
teams a shared vocabulary for describing components.

## Recommended Keys

| Key | Type | Purpose |
|---|---|---|
| `domain` | `str` | Problem area such as `motion`, `tracking`, `pose`, `visualization`, or `demo`. |
| `tags` | `list[str]` | Search/filter labels for catalogs and UI. |
| `input` | `str` | Expected public input contract, for example `FrameBuffer`, `Signal`, or `TwoDimGraphData`. |
| `output` | `str` | Public output contract returned by the component. |
| `params` | `dict` | JSON-friendly parameter descriptions. |
| `optional_extra` | `str` | Install extra required by the implementation, such as `opencv` or `visualization`. |
| `hardware` | `str` or `list[str]` | Runtime expectations such as `cpu`, `cuda`, or `camera`. |
| `realtime_safe` | `bool` | Whether the component is safe for realtime UI paths. |

## Example

```python
import sef
from sef.core.interfaces import StageCapabilities


@sef.analyzer(
    "vertical_velocity",
    description="Estimate vertical velocity from a tracked signal.",
    version="1.0.0",
    aliases=("velocity_y",),
    metadata={
        "domain": "motion",
        "tags": ["tracking", "kinematics"],
        "input": "Signal",
        "output": "TwoDimGraphData",
        "params": {
            "fps": {"type": "float", "default": 30.0, "min": 0.0},
        },
        "realtime_safe": True,
    },
    capabilities=StageCapabilities.streaming(stateful=False, realtime_safe=True),
)
def vertical_velocity(signal, fps: float = 30.0):
    ...
```

## Parameter Metadata

Keep parameter metadata simple and serializable. Recommended fields:

| Key | Purpose |
|---|---|
| `type` | Human-readable type such as `int`, `float`, `str`, `bool`, `tuple[int, int]`. |
| `default` | Default value when the parameter is optional. |
| `required` | Whether the user must pass the parameter. |
| `min` / `max` | Numeric bounds where meaningful. |
| `choices` | Allowed values for enum-like parameters. |
| `description` | Short parameter explanation. |

Example:

```python
metadata={
    "params": {
        "tracker_type": {
            "type": "str",
            "default": "MIL",
            "choices": ["MIL", "KCF", "CSRT"],
            "description": "OpenCV tracker constructor to use.",
        }
    }
}
```

## Rules

- Do not put non-serializable objects in metadata.
- Do not rely on metadata for runtime behavior; use constructor parameters,
  capabilities, and public contracts for behavior.
- Keep metadata descriptive and stable. Changing metadata should not break
  pipeline execution.
- Use `capabilities` for execution planning facts, not metadata.
