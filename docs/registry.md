# Plugin Registry

`PluginRegistry` is the production contract for resolving named pipeline
components. It is deliberately small, deterministic, and suitable for UI
catalogs.

## Registration

```python
registry.register(
    PluginCategory.ANALYZER,
    "sample_count",
    SampleCountAnalyzer,
    "Count signal samples.",
    version="1.0.0",
    aliases=("count_samples",),
    metadata={"tags": ["demo"], "owner": "docs"},
)
```

The factory must be callable. Names, aliases, and categories must be non-empty
identifier strings. Aliases cannot collide with canonical names or other
aliases in the same category.

## Lookup

```python
definition = registry.get(PluginCategory.ANALYZER, "sample_count")
instance = registry.create(PluginCategory.ANALYZER, "count_samples")
```

`get()` and `create()` accept canonical names and aliases.

## Inspection

```python
registry.available_names(PluginCategory.ANALYZER, include_aliases=True)
registry.categories()
registry.describe(PluginCategory.ANALYZER)
registry.snapshot()
```

`describe()` returns JSON-safe metadata suitable for UI catalog rendering.
`snapshot()` returns immutable category/name mappings for diagnostics.

## Errors

Invalid registrations raise `InvalidPluginRegistrationError`.

Duplicate names or aliases raise `DuplicatePluginRegistrationError`.

Unknown plugin lookup still raises `KeyError` so builder code can wrap it as
`PluginResolutionError` with config path context.

## Professional Registry Guidelines

Use stable names. Do not encode environment-specific details in names.

Use aliases for backwards compatibility after renames.

Use `version` for plugin implementation compatibility, not config schema
versioning.

Use metadata for UI tags, ownership, expected input shape, or hardware
requirements.

Do not register plugins lazily during pipeline execution. Build the registry at
application startup and pass it to builders.
