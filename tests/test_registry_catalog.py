from __future__ import annotations

import unittest

from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.plugins.PluginRegistry import PluginCategory, PluginRegistry
from ui.services.registry_catalog import build_registry_catalog, filter_registry_cards


class RegistryCatalogTests(unittest.TestCase):
    def test_catalog_derives_searchable_tags_and_metadata(self) -> None:
        def analyzer_factory():
            return object()

        analyzer_factory.capabilities = StageCapabilities.streaming(realtime_safe=True)
        registry = PluginRegistry()
        registry.register(
            PluginCategory.ANALYZER,
            "custom_motion",
            analyzer_factory,
            "Motion analyzer",
            aliases=("motion_alias",),
            metadata={
                "tags": ["motion", "lab"],
                "domain": "vibration",
                "optional_extra": "analysis",
            },
        )

        catalog = build_registry_catalog(registry)
        card = catalog.cards[0]

        self.assertEqual(card.name, "custom_motion")
        self.assertIn("motion", card.tags)
        self.assertIn("analysis", card.tags)
        self.assertIn("streaming", card.tags)
        self.assertIn("realtime-safe", card.tags)
        self.assertEqual(card.optional_extra, "analysis")
        self.assertTrue(card.capabilities["supports_streaming"])

        self.assertEqual(filter_registry_cards(catalog.cards, query="vibration"), (card,))
        self.assertEqual(filter_registry_cards(catalog.cards, tags=("motion", "streaming")), (card,))
        self.assertEqual(filter_registry_cards(catalog.cards, optional_extras=("analysis",)), (card,))
        self.assertEqual(filter_registry_cards(catalog.cards, capabilities=("supports_streaming",)), (card,))
        self.assertEqual(filter_registry_cards(catalog.cards, tags=("missing",)), ())


if __name__ == "__main__":
    unittest.main()
