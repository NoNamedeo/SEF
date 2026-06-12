from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from library.core.pipeline.PipelineErrors import ConfigSchemaError, ConfigVersionError

PIPELINE_CONFIG_VERSION_KEY = "schema_version"
CURRENT_PIPELINE_CONFIG_VERSION = "1.0"
SUPPORTED_PIPELINE_CONFIG_VERSIONS = (CURRENT_PIPELINE_CONFIG_VERSION,)


@dataclass(frozen=True, slots=True)
class PipelineConfigMigration:
    """
    Declarative migration step between two config schema versions.

    Migrations are intentionally pure: they receive a mapping and return a new
    mapping. This keeps future schema evolution testable and prevents builders
    from mutating user-provided configuration objects.
    """

    source_version: str
    target_version: str
    migrate: Callable[[Mapping[str, Any]], Mapping[str, Any]]

    def apply(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """
        Apply the migration to a config mapping.

        Parameters
        ----------
        config:
            Source config in `source_version` schema.

        Returns
        -------
        dict[str, Any]
            Migrated config with `schema_version` set to `target_version`.

        Raises
        ------
        ConfigVersionError
            If the migration function does not return a mapping.
        """
        migrated = self.migrate(config)
        if not isinstance(migrated, Mapping):
            raise ConfigVersionError(
                "Config migration must return a mapping.",
                version=self.source_version,
                supported_versions=(self.target_version,),
                path=PIPELINE_CONFIG_VERSION_KEY,
            )
        result = dict(migrated)
        result[PIPELINE_CONFIG_VERSION_KEY] = self.target_version
        return result


@dataclass(frozen=True, slots=True)
class VersionedPipelineConfig:
    """
    Normalized pipeline configuration plus schema-version metadata.

    Attributes
    ----------
    root:
        Canonical top-level config mapping after migration.
    pipeline:
        Canonical `pipeline` section consumed by `ConfigPipelineBuilder`.
    schema_version:
        Resolved public schema version.
    explicit_version:
        Whether the user supplied `schema_version`.
    applied_migrations:
        Ordered migration labels applied during normalization.
    """

    root: Mapping[str, Any]
    pipeline: Mapping[str, Any]
    schema_version: str
    explicit_version: bool
    applied_migrations: tuple[str, ...] = ()

    def source_config(self) -> dict[str, Any]:
        """Return a compact source config safe to store in `PipelineContext`."""
        return {
            PIPELINE_CONFIG_VERSION_KEY: self.schema_version,
            "pipeline": dict(self.pipeline),
        }


@dataclass(frozen=True, slots=True)
class PipelineConfigVersionManager:
    """
    Validate and normalize declarative pipeline config versions.

    The current implementation supports the first public schema, ``1.0``.
    The migration list is an explicit extension point for future schema changes
    without pushing version conditionals into ConfigPipelineBuilder.
    """

    current_version: str = CURRENT_PIPELINE_CONFIG_VERSION
    supported_versions: tuple[str, ...] = SUPPORTED_PIPELINE_CONFIG_VERSIONS
    migrations: tuple[PipelineConfigMigration, ...] = field(default_factory=tuple)

    def normalize(self, config: Mapping[str, Any]) -> VersionedPipelineConfig:
        """
        Validate, migrate, and canonicalize a pipeline config.

        Parameters
        ----------
        config:
            User-provided top-level configuration mapping.

        Returns
        -------
        VersionedPipelineConfig
            Canonical config plus version metadata.

        Raises
        ------
        ConfigSchemaError
            If the root or `pipeline` section has an invalid shape.
        ConfigVersionError
            If the explicit schema version is malformed or unsupported.
        """
        if not isinstance(config, Mapping):
            raise ConfigSchemaError("'config' must be a mapping.", path="config")

        root = dict(config)
        version, explicit_version = self._read_version(root)
        migrated_root, applied_migrations = self._migrate_to_current(root, version)
        pipeline = migrated_root.get("pipeline")
        if not isinstance(pipeline, Mapping):
            raise ConfigSchemaError("Missing required config section 'pipeline'.", path="pipeline")
        canonical_pipeline = self._canonical_pipeline(pipeline)

        canonical_root = dict(migrated_root)
        canonical_root[PIPELINE_CONFIG_VERSION_KEY] = self.current_version
        canonical_root["pipeline"] = canonical_pipeline
        return VersionedPipelineConfig(
            root=canonical_root,
            pipeline=canonical_pipeline,
            schema_version=self.current_version,
            explicit_version=explicit_version,
            applied_migrations=applied_migrations,
        )

    @staticmethod
    def _canonical_pipeline(pipeline: Mapping[str, Any]) -> dict[str, Any]:
        """
        Return the public canonical pipeline section for the current schema.

        Early SEF examples used ``frame_cleaners`` for frame preprocessing.
        The stable schema calls the same stage ``frame_processors`` because
        both single-frame and frame-buffer processors are supported.
        """
        canonical = dict(pipeline)
        legacy_frame_processors = canonical.pop("frame_cleaners", None)
        if legacy_frame_processors is not None and "frame_processors" not in canonical:
            canonical["frame_processors"] = legacy_frame_processors
        return canonical

    def _read_version(self, root: Mapping[str, Any]) -> tuple[str, bool]:
        raw_version = root.get(PIPELINE_CONFIG_VERSION_KEY)
        if raw_version is None:
            return self.current_version, False
        if not isinstance(raw_version, str) or not raw_version.strip():
            raise ConfigVersionError(
                "schema_version must be a non-empty string.",
                version=None,
                supported_versions=self.supported_versions,
                path=PIPELINE_CONFIG_VERSION_KEY,
            )
        return raw_version.strip(), True

    def _migrate_to_current(self, root: Mapping[str, Any], version: str) -> tuple[dict[str, Any], tuple[str, ...]]:
        if version == self.current_version:
            return dict(root), ()

        migrations_by_source = {migration.source_version: migration for migration in self.migrations}
        current = version
        migrated: dict[str, Any] = dict(root)
        applied: list[str] = []
        while current != self.current_version and current in migrations_by_source:
            migration = migrations_by_source[current]
            migrated = migration.apply(migrated)
            applied.append(f"{migration.source_version}->{migration.target_version}")
            current = migration.target_version

        if current != self.current_version:
            raise ConfigVersionError(
                f"Unsupported pipeline config schema_version '{version}'. "
                f"Supported versions: {', '.join(self.supported_versions)}.",
                version=version,
                supported_versions=self.supported_versions,
                path=PIPELINE_CONFIG_VERSION_KEY,
            )

        return migrated, tuple(applied)


DEFAULT_PIPELINE_CONFIG_VERSION_MANAGER = PipelineConfigVersionManager()


def normalize_pipeline_config(config: Mapping[str, Any]) -> VersionedPipelineConfig:
    """
    Normalize a user config using the default public schema-version policy.

    This helper is the public convenience entry point used by
    `ConfigPipelineBuilder`.
    """
    return DEFAULT_PIPELINE_CONFIG_VERSION_MANAGER.normalize(config)


__all__ = [
    "CURRENT_PIPELINE_CONFIG_VERSION",
    "DEFAULT_PIPELINE_CONFIG_VERSION_MANAGER",
    "PIPELINE_CONFIG_VERSION_KEY",
    "PipelineConfigMigration",
    "PipelineConfigVersionManager",
    "SUPPORTED_PIPELINE_CONFIG_VERSIONS",
    "VersionedPipelineConfig",
    "normalize_pipeline_config",
]
