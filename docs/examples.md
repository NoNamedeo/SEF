# Examples

Examples live in the repository-level `examples/` directory.

## Minimal Pipeline

Repository file: `examples/minimal_pipeline.py`

Run:

```bash
python examples/minimal_pipeline.py
```

Expected output:

```text
results: 1
artifacts: 1
sample_count: 3.0
summary: Sample count: 3.0
```

This example demonstrates:

- public package-level imports;
- custom plugin implementations;
- registry-backed config construction;
- explicit config `schema_version`;
- typed analyzer output;
- UI-agnostic `TextArtifact`;
- `PipelineOutputs` inspection.
