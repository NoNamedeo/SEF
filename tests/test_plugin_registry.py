from __future__ import annotations

import unittest

from library.core.plugins.PluginRegistry import PluginRegistry, create_builtin_registry
from library.frame_cleaners.ColorStabilizationFrameCleaner import ColorStabilizationFrameCleaner
from library.signal_cleaners.MovingAverageCleaner import MovingAverageCleaner


class PluginRegistryTests(unittest.TestCase):
    def test_builtin_registry_exposes_core_components(self):
        registry = create_builtin_registry()

        analyzer_names = {plugin.name for plugin in registry.list("analyzer")}
        self.assertIn("vertical_position", analyzer_names)

        cleaner = registry.create("signal_cleaner", "moving_average", window_size=5)
        self.assertIsInstance(cleaner, MovingAverageCleaner)

        frame_cleaner = registry.create("frame_cleaner", "color_stabilization")
        self.assertIsInstance(frame_cleaner, ColorStabilizationFrameCleaner)

    def test_register_rejects_duplicates(self):
        registry = PluginRegistry()
        registry.register("analyzer", "test", lambda: object())

        with self.assertRaises(ValueError):
            registry.register("analyzer", "test", lambda: object())


if __name__ == "__main__":
    unittest.main()
