from __future__ import annotations

import unittest

from ui.models.pipeline_builder import (
    IntermediateFrameConfiguration,
    PipelineConfiguration,
    PluginConfig,
    VisualizerConfig,
)
from ui.services.pipeline_builder_service import suggested_visualizer_target_indices


class PipelineBuilderConfigurationTests(unittest.TestCase):
    def test_intermediate_visualizers_are_serialized_in_dedicated_section(self) -> None:
        config = PipelineConfiguration(
            frame_extractor=PluginConfig("opencv_buffered"),
            signal_extractor=PluginConfig("opencv_tracker"),
            analyzers=(PluginConfig("vertical_position"),),
            visualizers=(VisualizerConfig("matplotlib_function", result_indices=(0,)),),
            intermediate_frames=IntermediateFrameConfiguration(
                enabled=True,
                max_stored_frames=30,
                visualizers=(PluginConfig("intermediate_frames_grid"),),
            ),
        ).to_dict()

        pipeline = config["pipeline"]

        self.assertEqual(
            pipeline["visualizers"],
            [{"name": "matplotlib_function", "result_indices": [0]}],
        )
        self.assertEqual(
            pipeline["intermediate_frames"]["visualizers"],
            [{"name": "intermediate_frames_grid"}],
        )

    def test_intermediate_visualizers_do_not_target_analyzer_results(self) -> None:
        self.assertIsNone(
            suggested_visualizer_target_indices(
                "intermediate_frames",
                ("vertical_position",),
            )
        )


if __name__ == "__main__":
    unittest.main()
