# Examples

Examples live in the repository-level `examples/` directory.

## Minimal Pipeline

Repository file: `examples/minimal_pipeline.py`

Run:

```bash
python -m examples.minimal_pipeline
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
- facade-backed pipeline construction;
- typed analyzer output;
- UI-agnostic `TextArtifact`;
- `PipelineOutputs` inspection.
