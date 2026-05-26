# Versioning Policy

SEF has multiple version axes. They should not be mixed.

## Library Version

The Python package version describes installable library compatibility.

Use semantic versioning:

- patch: bug fixes with no contract changes;
- minor: compatible public API additions;
- major: breaking public API or config changes without automatic migration.

## Config Schema Version

Config schema version is stored in top-level `schema_version`.

Current version: `1.0`.

Unversioned configs are interpreted as `1.0` for compatibility.

New schema versions must use `PipelineConfigMigration` to keep older configs
loadable when practical.

## Plugin Version

Plugin version is stored in `PluginDefinition.version`. It describes the plugin
implementation, not the config schema.

Use plugin aliases to preserve old names after a rename.

Use metadata to describe compatibility details, for example:

```python
metadata={
    "input": "COCO pose signal",
    "output": "COCOPoseTennisFrameData",
    "realtime_safe": True,
}
```

## Deprecation Policy

Before removing a public contract:

1. add a replacement;
2. document the replacement;
3. keep the old name as an alias or adapter when feasible;
4. add a deprecation note;
5. remove only in a major version.

Config field removals should be implemented as migrations whenever possible.
