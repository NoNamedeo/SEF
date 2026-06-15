from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from sef.cli.command_handlers import (
    config_schema,
    doctor,
    init_project,
    inspect_component,
    list_components,
    run_pipeline,
    validate_pipeline,
    version,
)
from sef.cli.diagnostics import CliDiagnostics
from sef.core.plugins import PluginCategory


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SEF command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:  # noqa: BLE001 - CLI boundary converts all failures to readable errors.
        debug = bool(getattr(args, "debug", False))
        CliDiagnostics.from_exception(exc).print(debug=debug)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level parser and wire subcommands to handlers."""
    parser = argparse.ArgumentParser(
        prog="sef",
        description="Run, validate, inspect, and scaffold SEF pipeline projects.",
    )
    parser.add_argument("--debug", action="store_true", help="Print full tracebacks for CLI errors.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    _add_init_parser(subcommands)
    _add_doctor_parser(subcommands)
    _add_run_parser(subcommands)
    _add_validate_parser(subcommands)
    _add_components_parser(subcommands)
    _add_version_parser(subcommands)
    _add_config_parser(subcommands)

    return parser


def _add_init_parser(subcommands: Any) -> None:
    init_parser = subcommands.add_parser("init", help="Create a SEF project scaffold.")
    init_parser.add_argument(
        "template",
        nargs="?",
        choices=("tracking-demo", "plugin"),
        help="Optional scaffold template. Supports tracking-demo and plugin.",
    )
    init_parser.add_argument("--force", action="store_true", help="Overwrite SEF scaffold files created by sef init.")
    init_parser.set_defaults(handler=init_project)


def _add_doctor_parser(subcommands: Any) -> None:
    doctor_parser = subcommands.add_parser("doctor", help="Inspect the local SEF environment.")
    doctor_parser.add_argument("--config", default=None, help="Optional pipeline config to validate during checks.")
    doctor_parser.set_defaults(handler=doctor)


def _add_run_parser(subcommands: Any) -> None:
    run_parser = subcommands.add_parser("run", help="Run a YAML or JSON pipeline config.")
    run_parser.add_argument("config", help="Path to a .yaml, .yml, or .json pipeline config.")
    run_parser.add_argument("--pipeline-id", help="Optional pipeline id for metadata and artifacts.")
    run_parser.add_argument("--dry-run", action="store_true", help="Build and explain the pipeline without executing it.")
    run_parser.add_argument("--explain", action="store_true", help="Print the execution plan.")
    run_parser.add_argument("--output", help="Directory for run summaries, plans, normalized config, and artifacts.")
    run_parser.add_argument("--debug", action="store_true", help="Print full tracebacks for CLI errors.")
    run_parser.set_defaults(handler=run_pipeline)


def _add_validate_parser(subcommands: Any) -> None:
    validate_parser = subcommands.add_parser("validate", help="Validate a YAML or JSON pipeline config.")
    validate_parser.add_argument("config", help="Path to a .yaml, .yml, or .json pipeline config.")
    validate_parser.add_argument("--strict", action="store_true", help="Treat unknown config fields as errors.")
    validate_parser.add_argument("--debug", action="store_true", help="Print full tracebacks for CLI errors.")
    validate_parser.set_defaults(handler=validate_pipeline)


def _add_components_parser(subcommands: Any) -> None:
    components_parser = subcommands.add_parser("components", help="Inspect registered SEF components.")
    component_subcommands = components_parser.add_subparsers(dest="components_command", required=True)

    list_parser = component_subcommands.add_parser("list", help="List registered component plugins.")
    list_parser.add_argument(
        "--category",
        choices=tuple(category.value for category in PluginCategory),
        help="Filter components by plugin category.",
    )
    list_parser.add_argument("--json", action="store_true", help="Print component descriptors as JSON.")
    list_parser.set_defaults(handler=list_components)

    inspect_parser = component_subcommands.add_parser("inspect", help="Inspect one component in detail.")
    inspect_parser.add_argument("name", help="Component name or alias to inspect.")
    inspect_parser.add_argument(
        "--category",
        choices=tuple(category.value for category in PluginCategory),
        help="Disambiguate components that share a name across categories.",
    )
    inspect_parser.add_argument("--json", action="store_true", help="Print the inspection payload as JSON.")
    inspect_parser.set_defaults(handler=inspect_component)


def _add_version_parser(subcommands: Any) -> None:
    version_parser = subcommands.add_parser("version", help="Print SEF version and installation details.")
    version_parser.set_defaults(handler=version)


def _add_config_parser(subcommands: Any) -> None:
    config_parser = subcommands.add_parser("config", help="Inspect SEF public config helpers.")
    config_subcommands = config_parser.add_subparsers(dest="config_command", required=True)
    schema_parser = config_subcommands.add_parser("schema", help="Print the practical public config schema.")
    schema_parser.add_argument("--format", choices=("json", "yaml"), default="json", help="Output format.")
    schema_parser.set_defaults(handler=config_schema)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
