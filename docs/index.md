# SEF Core Documentation

This documentation describes the public SEF core contracts for users who build
pipeline integrations, write plugins, expose APIs, or maintain UI adapters.

SEF core is the stable layer of the project. It owns pipeline orchestration,
typed artifacts, plugin resolution, config versioning, streaming decisions,
runtime errors, and UI-agnostic output contracts. Concrete implementations such
as OpenCV extractors, YOLO pose components, Matplotlib visualizers, and
Streamlit views are adapters around this core.

## Documentation Map

- [Overview](overview.md): architecture and responsibility boundaries.
- [Getting Started](getting-started.md): a minimal runnable pipeline.
- [Public API](public-api.md): stable import surfaces and compatibility rules.
- [Configuration](configuration.md): versioned config schema reference.
- [Registry](registry.md): plugin registration, aliases, metadata, snapshots.
- [Plugin Authoring](plugin-authoring.md): how to implement new components.
- [Streaming Runtime](streaming-runtime.md): streaming contracts, planner, buffers.
- [Error Handling](error-handling.md): typed errors and handling patterns.
- [Versioning](versioning.md): package, config, plugin, and deprecation policy.
- [Examples](examples.md): runnable example modules and expected outputs.
- [Reference: Component Contracts](reference/component-contracts.md)
- [Reference: Data and Artifacts](reference/data-and-artifacts.md)
- [Reference: Pipeline Runtime](reference/pipeline-runtime.md)
- [Reference: Error Model](reference/errors.md)

The older single-page contract reference remains available at
[Public Contracts](public-contracts.md), but new documentation should link to
the modular pages above.

## Build the Documentation Site

Install the optional docs dependencies and run MkDocs from the repository root:

```bash
pip install -e ".[docs]"
mkdocs serve
```

The site navigation is defined in the repository-level `mkdocs.yml`.

## Contract Scope

Public contracts are the names exported from public package initializers:

- `sef`
- `sef.core`
- `sef.core.artifacts`
- `sef.core.events`
- `sef.core.interfaces`
- `sef.core.interfaces.pipeline`
- `sef.core.pipeline`
- `sef.core.plugins`
- `sef.core.realtime`
- `sef.core.visualization`

External code should prefer package-level imports over direct module paths.
Direct module paths may continue to work, but package exports define the
supported public surface.

## Maintainer Standard

Any public contract change should include:

1. a documentation update;
2. a focused test update;
3. an explicit compatibility decision;
4. migration guidance when existing configs or plugins are affected.
