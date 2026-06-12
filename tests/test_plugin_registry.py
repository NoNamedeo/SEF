from __future__ import annotations

import unittest

from sef.builtin.frame_processors.ColorStabilizationFrameProcessor import ColorStabilizationFrameProcessor
from sef.builtin.signal_cleaners.single_tracker.MovingAverageCleaner import MovingAverageCleaner
from sef.core.errors import DuplicatePluginRegistrationError, InvalidPluginRegistrationError
from sef.builtin.registry import create_builtin_registry
from sef.core.plugins.PluginRegistry import PluginDefinition, PluginRegistry


class RegistryPlugin:
    pass


class PluginRegistryTests(unittest.TestCase):
    def test_builtin_registry_exposes_core_components(self):
        registry = create_builtin_registry()

        analyzer_names = {plugin.name for plugin in registry.list("analyzer")}
        self.assertIn("vertical_position", analyzer_names)

        cleaner = registry.create("signal_cleaner", "moving_average", window_size=5)
        self.assertIsInstance(cleaner, MovingAverageCleaner)

        single_frame_processor = registry.create("single_frame_processor", "color_stabilization")
        self.assertIsInstance(single_frame_processor, ColorStabilizationFrameProcessor)

    def test_register_rejects_duplicates(self):
        registry = PluginRegistry()
        registry.register("analyzer", "test", lambda: object())

        with self.assertRaises(DuplicatePluginRegistrationError):
            registry.register("analyzer", "test", lambda: object())

    def test_registry_resolves_aliases_and_exposes_descriptors(self):
        registry = PluginRegistry()

        definition = registry.register(
            "analyzer",
            "canonical",
            RegistryPlugin,
            "Example plugin.",
            version="2.1.0",
            aliases=("alias",),
            metadata={"owner": "core"},
        )

        self.assertIs(registry.get("analyzer", "canonical"), definition)
        self.assertIs(registry.get("analyzer", "alias"), definition)
        self.assertIsInstance(registry.create("analyzer", "alias"), RegistryPlugin)
        self.assertEqual(registry.available_names("analyzer", include_aliases=True), ("alias", "canonical"))
        self.assertEqual(registry.categories(), ("analyzer",))

        descriptor = registry.describe("analyzer")[0]
        self.assertEqual(descriptor["name"], "canonical")
        self.assertEqual(descriptor["version"], "2.1.0")
        self.assertEqual(descriptor["aliases"], ["alias"])
        self.assertEqual(descriptor["metadata"], {"owner": "core"})

        with self.assertRaises(TypeError):
            definition.metadata["owner"] = "changed"

    def test_registry_snapshot_is_immutable(self):
        registry = PluginRegistry()
        registry.register("analyzer", "canonical", RegistryPlugin)

        snapshot = registry.snapshot()

        with self.assertRaises(TypeError):
            snapshot["analyzer"] = {}
        with self.assertRaises(TypeError):
            snapshot["analyzer"]["canonical"] = PluginDefinition("analyzer", "other", RegistryPlugin)

    def test_register_rejects_invalid_definitions(self):
        registry = PluginRegistry()

        with self.assertRaises(InvalidPluginRegistrationError):
            registry.register("analyzer", "bad name", RegistryPlugin)
        with self.assertRaises(InvalidPluginRegistrationError):
            registry.register("analyzer", "not_callable", object())
        with self.assertRaises(InvalidPluginRegistrationError):
            registry.register("analyzer", "bad_aliases", RegistryPlugin, aliases="alias")
        with self.assertRaises(InvalidPluginRegistrationError):
            registry.register("analyzer", "bad_metadata", RegistryPlugin, metadata=object())

    def test_register_rejects_alias_collisions(self):
        registry = PluginRegistry()
        registry.register("analyzer", "canonical", RegistryPlugin, aliases=("alias",))

        with self.assertRaises(DuplicatePluginRegistrationError):
            registry.register("analyzer", "other", RegistryPlugin, aliases=("alias",))
        with self.assertRaises(DuplicatePluginRegistrationError):
            registry.register("analyzer", "alias", RegistryPlugin)


if __name__ == "__main__":
    unittest.main()
