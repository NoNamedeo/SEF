from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from library.core.artifacts.MaskArtifacts import (
    FrameMaskArtifact,
    IntermediateFrameArtifact,
    MotionMaskArtifact,
    ProtectedRegionArtifact,
    TargetMaskArtifact,
)
from library.core.artifacts.MaskOperations import (
    intersect_masks,
    merge_masks,
    subtract_masks,
    validate_binary_mask,
)


def test_frame_mask_artifact_normalizes_and_serializes_debug_state() -> None:
    source_mask = np.array([[0, 255], [1, 0]], dtype=np.uint8)

    artifact = FrameMaskArtifact(
        mask=source_mask,
        frame_index=7,
        timestamp_seconds=0.25,
        label="foreground",
        metadata={"source": "unit-test"},
        config={"threshold": 0.8},
    )

    source_mask[0, 0] = 255

    assert artifact.shape == (2, 2)
    assert artifact.height == 2
    assert artifact.width == 2
    assert artifact.active_pixel_count == 2
    assert artifact.coverage_ratio == 0.5
    assert artifact.mask.flags.writeable is False
    assert artifact.as_bool_array(copy=False)[0, 0] == np.False_
    assert artifact.as_uint8_array().tolist() == [[0, 255], [255, 0]]

    payload = artifact.to_dict(include_mask=True)
    assert payload["artifact_type"] == "frame_mask"
    assert payload["metadata"] == {"source": "unit-test"}
    assert payload["config"] == {"threshold": 0.8}
    assert payload["mask"] == [[0, 1], [1, 0]]
    assert "FrameMaskArtifact" in artifact.debug_string()
    assert json.loads(artifact.to_json())["artifact_type"] == "frame_mask"

    with pytest.raises(FrozenInstanceError):
        artifact.frame_index = 8  # type: ignore[misc]
    with pytest.raises(ValueError):
        artifact.as_bool_array(copy=False)[0, 0] = True
    with pytest.raises(TypeError):
        artifact.metadata["new"] = "value"  # type: ignore[index]


def test_mask_operations_validate_and_combine_compatible_masks() -> None:
    first = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    second = np.array([[False, True], [True, True]], dtype=np.bool_)

    validate_binary_mask(first)

    assert merge_masks(first, second).tolist() == [[True, True], [True, True]]
    assert intersect_masks(first, second).tolist() == [[False, False], [False, True]]
    assert subtract_masks(first, second).tolist() == [[True, False], [False, False]]

    with pytest.raises(ValueError, match="binary values"):
        validate_binary_mask(np.array([[2]], dtype=np.uint8))
    with pytest.raises(ValueError, match="spatial shape"):
        merge_masks(first, np.ones((3, 3), dtype=np.bool_))


def test_target_and_protected_region_artifacts_preserve_typed_fields() -> None:
    mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    motion = MotionMaskArtifact(mask=mask, label="moving-pixels")
    target = TargetMaskArtifact(mask=mask, target_id="subject-1")
    protected_region = ProtectedRegionArtifact(mask=mask, region_id="manual-roi", reason="operator override")

    assert motion.to_dict()["artifact_type"] == "motion_mask"
    assert target.to_dict()["artifact_type"] == "target_mask"
    assert target.to_dict()["target_id"] == "subject-1"
    assert protected_region.to_dict()["artifact_type"] == "protected_region"
    assert protected_region.to_dict()["region_id"] == "manual-roi"
    assert protected_region.to_dict()["reason"] == "operator override"

    with pytest.raises(ValueError, match="target_id"):
        TargetMaskArtifact(mask=mask, target_id=" ")
    with pytest.raises(ValueError, match="reason"):
        ProtectedRegionArtifact(mask=mask, reason=" ")


def test_intermediate_frame_artifact_copies_and_validates_snapshots() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 1] = (10, 20, 30)

    artifact = IntermediateFrameArtifact(
        image=image,
        stage_name="after_equalization",
        frame_index=3,
        color_space="BGR",
        metadata={"cleaner": "histogram"},
        config={"clip_limit": 2.0},
    )

    image[0, 1] = (255, 255, 255)

    assert artifact.shape == (2, 3, 3)
    assert artifact.spatial_shape == (2, 3)
    assert artifact.channels == 3
    assert artifact.dtype == np.dtype("uint8")
    assert artifact.frame.flags.writeable is False
    assert artifact.as_array(copy=False)[0, 1].tolist() == [10, 20, 30]
    assert artifact.to_dict(include_image=True)["image"][0][1] == [10, 20, 30]
    assert "after_equalization" in artifact.debug_string()

    artifact.ensure_mask_compatible(FrameMaskArtifact(mask=np.zeros((2, 3), dtype=np.bool_)))
    with pytest.raises(ValueError, match="spatial shape"):
        artifact.ensure_mask_compatible(np.zeros((3, 2), dtype=np.bool_))
    with pytest.raises(ValueError, match="stage_name"):
        IntermediateFrameArtifact(image=np.zeros((2, 3), dtype=np.uint8), stage_name=" ")
