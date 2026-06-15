"""UI catalog adapter for the SEF plugin registry."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping

from sef.core.plugins.PluginRegistry import PluginCategory, PluginDefinition, PluginRegistry
from ui.services.plugin_display import plugin_capabilities_label


@dataclass(frozen=True, slots=True)
class RegistryPluginCard:
    """Presentation model for one registry plugin definition."""

    name: str
    category: str
    category_label: str
    version: str
    aliases: tuple[str, ...]
    factory_path: str
    description: str
    metadata: Mapping[str, Any]
    optional_extra: str | None
    tags: tuple[str, ...]
    capabilities: Mapping[str, bool]
    capability_label: str
    search_text: str

    def table_row(self) -> dict[str, str]:
        """Return the compact table representation used by the registry tab."""
        return {
            "name": self.name,
            "category": self.category_label,
            "version": self.version,
            "aliases": ", ".join(self.aliases) or "-",
            "tags": ", ".join(self.tags) or "-",
            "optional_extra": self.optional_extra or "-",
            "capabilities": self.capability_label,
            "factory": self.factory_path,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class RegistryCatalog:
    """Derived registry catalog plus reusable filter options."""

    cards: tuple[RegistryPluginCard, ...]
    categories: tuple[str, ...]
    tags: tuple[str, ...]
    optional_extras: tuple[str, ...]
    capability_names: tuple[str, ...]


def build_registry_catalog(registry: PluginRegistry) -> RegistryCatalog:
    """Build a deterministic presentation catalog from the live registry."""
    cards = tuple(
        sorted(
            (_build_card(definition) for definition in registry.list()),
            key=lambda card: (card.category, card.name),
        )
    )
    return RegistryCatalog(
        cards=cards,
        categories=tuple(sorted({card.category for card in cards})),
        tags=tuple(sorted({tag for card in cards for tag in card.tags})),
        optional_extras=tuple(sorted({card.optional_extra for card in cards if card.optional_extra})),
        capability_names=tuple(sorted({name for card in cards for name in card.capabilities})),
    )


def filter_registry_cards(
    cards: Iterable[RegistryPluginCard],
    *,
    query: str = "",
    categories: Iterable[str] = (),
    tags: Iterable[str] = (),
    optional_extras: Iterable[str] = (),
    capabilities: Iterable[str] = (),
) -> tuple[RegistryPluginCard, ...]:
    """Return cards matching all selected registry filters."""
    normalized_query = query.strip().lower()
    selected_categories = set(categories)
    selected_tags = set(tags)
    selected_extras = set(optional_extras)
    selected_capabilities = set(capabilities)

    filtered = []
    for card in cards:
        if selected_categories and card.category not in selected_categories:
            continue
        if selected_tags and not selected_tags.issubset(set(card.tags)):
            continue
        if selected_extras and card.optional_extra not in selected_extras:
            continue
        if selected_capabilities and not all(card.capabilities.get(name) is True for name in selected_capabilities):
            continue
        if normalized_query and normalized_query not in card.search_text:
            continue
        filtered.append(card)
    return tuple(filtered)


def registry_category_label(category: str | PluginCategory) -> str:
    """Return a human-readable category label."""
    return str(category).replace("_", " ").title()


def _build_card(definition: PluginDefinition) -> RegistryPluginCard:
    descriptor = definition.as_dict()
    metadata = dict(descriptor.get("metadata", {}) or {})
    capabilities = _capabilities_dict(definition)
    tags = _derive_tags(definition, metadata, capabilities)
    optional_extra = _optional_extra(metadata)
    values_for_search = [
        descriptor.get("name", ""),
        descriptor.get("category", ""),
        descriptor.get("description", ""),
        descriptor.get("version", ""),
        descriptor.get("factory_path", ""),
        optional_extra or "",
        *descriptor.get("aliases", []),
        *tags,
        *_flatten_metadata(metadata),
        *capabilities.keys(),
    ]
    return RegistryPluginCard(
        name=str(descriptor["name"]),
        category=str(descriptor["category"]),
        category_label=registry_category_label(str(descriptor["category"])),
        version=str(descriptor["version"]),
        aliases=tuple(str(alias) for alias in descriptor.get("aliases", [])),
        factory_path=str(descriptor["factory_path"]),
        description=str(descriptor.get("description", "")),
        metadata=metadata,
        optional_extra=optional_extra,
        tags=tags,
        capabilities=capabilities,
        capability_label=plugin_capabilities_label(definition),
        search_text=" ".join(str(value).lower() for value in values_for_search if value),
    )


def _capabilities_dict(definition: PluginDefinition) -> dict[str, bool]:
    capabilities = getattr(definition.factory, "capabilities", None)
    as_dict = getattr(capabilities, "as_dict", None)
    if callable(as_dict):
        return {str(key): bool(value) for key, value in as_dict().items()}
    return {}


def _derive_tags(
    definition: PluginDefinition,
    metadata: Mapping[str, Any],
    capabilities: Mapping[str, bool],
) -> tuple[str, ...]:
    tags = set(_metadata_tags(metadata))
    tags.add(str(definition.category))
    tags.add(_category_role_tag(str(definition.category)))
    if definition.factory_path.startswith("sef.builtin."):
        tags.add("builtin")
    optional_extra = _optional_extra(metadata)
    if optional_extra:
        tags.add(optional_extra)
        tags.add("optional-dependency")
    if capabilities.get("supports_streaming"):
        tags.add("streaming")
    if capabilities.get("requires_complete_sequence"):
        tags.add("batch")
    if capabilities.get("realtime_safe"):
        tags.add("realtime-safe")
    if capabilities.get("supports_frame_parallelism"):
        tags.add("parallel")
    return tuple(sorted(tags))


def _metadata_tags(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    raw_tags = metadata.get("tags", ())
    if isinstance(raw_tags, str):
        return (raw_tags,)
    if isinstance(raw_tags, Iterable):
        return tuple(str(tag) for tag in raw_tags if str(tag).strip())
    return ()


def _optional_extra(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("optional_extra")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _category_role_tag(category: str) -> str:
    if "processor" in category:
        return "processing"
    if category in {PluginCategory.FRAME_EXTRACTOR.value, PluginCategory.SIGNAL_EXTRACTOR.value}:
        return "input"
    if category == PluginCategory.VISUALIZER.value:
        return "output"
    if category == PluginCategory.ANALYZER.value:
        return "analysis"
    if category == PluginCategory.BRANCHING_RULE.value:
        return "orchestration"
    return "pipeline"


def _flatten_metadata(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key, value in metadata.items():
        values.append(str(key))
        if isinstance(value, Mapping):
            values.extend(_flatten_metadata(value))
        elif isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return tuple(values)


def metadata_as_pretty_json(metadata: Mapping[str, Any]) -> str:
    """Return stable JSON for metadata preview/editing."""
    return json.dumps(dict(metadata), indent=2, sort_keys=True, default=str)
