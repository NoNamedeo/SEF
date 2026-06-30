from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import sef
from sef.api import from_config
from sef.cli.artifact_writer import ArtifactWriter
from sef.cli.component_inspection import (
    component_descriptors,
    component_inspection,
    find_component_matches,
    print_component_inspection,
)
from sef.cli.config_io import dump_config_schema, inspect_cli_config, load_config_with_raw, public_config_schema
from sef.cli.diagnostics import CliDiagnostics
from sef.cli.environment_checks import (
    doctor_dependencies,
    doctor_installation,
    doctor_matplotlib_cache,
    doctor_opencv_trackers,
    doctor_project_directories,
    doctor_python,
    installation_mode,
    sef_distribution_version,
)
from sef.cli.output import print_info, print_ok
from sef.cli.registry_loader import CliRegistryLoader
from sef.cli.scaffold import ProjectScaffolder
from sef.core.pipeline.PipelineExecutionPlan import PipelineExecutionPlan
from sef.core.pipeline.PipelineRunOptions import (
    PipelineExecutionPlanLevel,
)


def init_project(args: argparse.Namespace) -> int:
    """Create a SEF project scaffold."""
    root = Path.cwd()
    template = args.template or "default"
    created, skipped = ProjectScaffolder(root).create(template=template, force=bool(args.force))

    print(f"initialized: {root}")
    print(f"template: {template}")
    for path in created:
        print(f"created: {path.relative_to(root)}")
    for path in skipped:
        print(f"skipped: {path.relative_to(root)}")
    if skipped and not args.force:
        print("hint: use --force to overwrite SEF scaffold files that contain the scaffold marker.")
    return 0


def doctor(args: argparse.Namespace) -> int:
    """Inspect the local SEF environment."""
    diagnostics = CliDiagnostics()
    print_info("doctor")
    doctor_python(diagnostics)
    doctor_installation()
    doctor_dependencies(diagnostics)
    doctor_opencv_trackers(diagnostics)
    doctor_matplotlib_cache(diagnostics)
    doctor_project_directories(diagnostics)

    registry, import_result = CliRegistryLoader().load()
    diagnostics.extend(import_result.diagnostics)
    print_ok(f"registered_components={len(registry.list())}")
    print_ok(f"local_plugin_modules={len(import_result.loaded_paths)}")

    if args.config:
        try:
            raw_config, config = load_config_with_raw(args.config)
            inspection = inspect_cli_config(raw_config, config, strict=False)
            diagnostics.extend(inspection.warnings)
            diagnostics.extend(inspection.errors)
            from_config(config, registry=registry).build_context()
            print_ok(f"config_valid={args.config}")
        except Exception as exc:  # noqa: BLE001 - doctor keeps scanning but exits non-zero for blockers.
            diagnostics.add_error(
                f"Config check failed for {args.config}.",
                cause=str(exc),
                suggestion="Fix the config path/schema or run `sef validate <config> --strict`.",
            )

    diagnostics.print()
    return diagnostics.exit_code()


def run_pipeline(args: argparse.Namespace) -> int:
    """Run or explain a YAML/JSON pipeline config."""
    registry, import_result = CliRegistryLoader().load()
    diagnostics = CliDiagnostics(list(import_result.diagnostics))
    raw_config, config = load_config_with_raw(args.config)
    inspection = inspect_cli_config(raw_config, config, strict=False)
    diagnostics.extend(inspection.warnings)
    diagnostics.extend(inspection.errors)
    if diagnostics.has_errors:
        diagnostics.print()
        return diagnostics.exit_code()

    facade = from_config(config, pipeline_id=args.pipeline_id, registry=registry)
    plan = facade.execution_plan() if args.explain else None
    if args.explain:
        if plan is None:
            raise RuntimeError("Execution plan was not built for --explain.")
        print(plan.as_text())
        print_materialization_boundaries(plan)

    if args.dry_run:
        print(f"dry_run: {args.config}")
        print(f"schema_version: {config.get('schema_version')}")
        print(f"registered_components: {len(registry.list())}")
        if args.output:
            writer = ArtifactWriter(args.output)
            writer.write(outputs=None, config=config, execution_plan=plan, dry_run=True)
            diagnostics.extend(writer.warnings)
            print(f"output: {Path(args.output).resolve()}")
        diagnostics.print()
        return diagnostics.exit_code()

    run_options = facade.configured_run_options().with_required(
        execution_plan=PipelineExecutionPlanLevel.FULL if args.explain else None,
    )
    outputs = facade.run(
        run_options=run_options,
    )
    print(f"pipeline_id: {outputs.metadata.pipeline_id}")
    print(f"results: {len(outputs.results)}")
    print(f"artifacts: {outputs.artifact_count}")

    if args.output:
        writer = ArtifactWriter(args.output)
        writer.write(outputs=outputs, config=config, execution_plan=plan, dry_run=False)
        diagnostics.extend(writer.warnings)
        print(f"output: {Path(args.output).resolve()}")

    diagnostics.print()
    return diagnostics.exit_code()


def validate_pipeline(args: argparse.Namespace) -> int:
    """Validate a YAML/JSON pipeline config."""
    registry, import_result = CliRegistryLoader().load()
    diagnostics = CliDiagnostics(list(import_result.diagnostics))
    raw_config, config = load_config_with_raw(args.config)
    inspection = inspect_cli_config(raw_config, config, strict=bool(args.strict))
    diagnostics.extend(inspection.warnings)
    diagnostics.extend(inspection.errors)
    if diagnostics.has_errors:
        diagnostics.print()
        return diagnostics.exit_code()

    context = from_config(config, registry=registry).build_context()
    print(f"valid: {args.config}")
    print(f"schema_version: {context.source_config.get('schema_version')}")
    print(f"analyzers: {len(context.analyzers)}")
    print(f"visualizers: {len(context.visualizers) + len(context.visualizer_bindings)}")
    diagnostics.print()
    return diagnostics.exit_code()


def list_components(args: argparse.Namespace) -> int:
    """List registered component plugins."""
    registry, import_result = CliRegistryLoader().load()
    descriptors = component_descriptors(registry.describe(args.category))
    diagnostics = CliDiagnostics(list(import_result.diagnostics))
    if args.json:
        print(json.dumps(descriptors, indent=2, sort_keys=True))
        diagnostics.print()
        return diagnostics.exit_code()

    if not descriptors:
        print("No components registered.")
        diagnostics.print()
        return diagnostics.exit_code()

    for descriptor in descriptors:
        description = descriptor.get("description") or "-"
        print(f"{descriptor['category']:<24} {descriptor['name']:<28} {description}")
    diagnostics.print()
    return diagnostics.exit_code()


def inspect_component(args: argparse.Namespace) -> int:
    """Inspect one registered component plugin."""
    registry, import_result = CliRegistryLoader().load()
    diagnostics = CliDiagnostics(list(import_result.diagnostics))
    matches = find_component_matches(registry, args.name, category=args.category)
    if not matches:
        diagnostics.add_error(
            f"Component `{args.name}` was not found.",
            cause="No registered builtin or local plugin matches that name or alias.",
            suggestion="Run `sef components list` or check local plugin import warnings.",
        )
        diagnostics.print()
        return 1
    if len(matches) > 1:
        categories = ", ".join(sorted(definition.category for definition in matches))
        diagnostics.add_error(
            f"Component `{args.name}` is ambiguous across categories: {categories}.",
            suggestion="Re-run with `--category <category>`.",
        )
        diagnostics.print()
        return 1

    payload = component_inspection(matches[0])
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_component_inspection(payload)
    diagnostics.print()
    return diagnostics.exit_code()


def version(_args: argparse.Namespace) -> int:
    """Print SEF version and installation details."""
    print(f"sef: {sef_distribution_version()}")
    print(f"python: {platform.python_version()} ({sys.executable})")
    print(f"install_path: {Path(sef.__file__).resolve()}")
    print(f"mode: {installation_mode()}")
    return 0


def config_schema(args: argparse.Namespace) -> int:
    """Print the practical public config schema."""
    print(dump_config_schema(public_config_schema(), output_format=args.format), end="" if args.format == "yaml" else "\n")
    return 0


def print_materialization_boundaries(plan: PipelineExecutionPlan) -> None:
    """Print pipeline plan materialization boundaries."""
    if not plan.materialization_boundaries:
        print("materialization_boundaries: none")
        return
    print("materialization_boundaries:")
    for stage in plan.materialization_boundaries:
        print(f"- {stage.stage_id}: {stage.component_name}")
