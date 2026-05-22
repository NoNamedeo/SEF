from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from library.core.pose.COCOSkeletonNormalizer import (
    COCOSkeletonNormalizationConfig,
    COCOSkeletonNormalizer,
)
from library.Trainer_class_COCOSkeletonTennis import Trainer_class_COCOSkeletonTennis

# ----------------------------
# LABELS
# ----------------------------

LABEL_MAP = {
    "backhand": 0,
    "forehand": 1,
    "ready_position": 2,
    "serve": 3,
}

# ----------------------------
# PATHS (relativi al progetto)
# ----------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "datasets" / "COCO_skeleton_tennis"

OUTPUT_FILE = DATASET_DIR / "COCO_skeleton_tennis_for_training.json"

# ----------------------------
# MODEL LOAD (coerente col tuo extractor)
# ----------------------------

def load_model():
    model_path = BASE_DIR / "library" / "signal_extractors" / "YOLOPoseModels" / "yolo11s-pose.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found at {model_path}. "
            "Ensure it matches your extractor setup."
        )

    return YOLO(str(model_path))


# ----------------------------
# PARSE YOLO RESULT
# ----------------------------

def parse_skeleton(result):
    if result.keypoints is None:
        return np.zeros((17, 2)), np.zeros(17)

    kpts = result.keypoints

    if kpts.xy.shape[0] == 0:
        return np.zeros((17, 2)), np.zeros(17)

    xy = kpts.xy[0].cpu().numpy()
    conf = (
        np.ones(17)
        if kpts.conf is None
        else kpts.conf[0].cpu().numpy()
    )

    return xy.tolist(), conf.tolist()


# ----------------------------
# PROCESS SINGLE IMAGE
# ----------------------------

def process_image(model, image_path: Path):
    result = model(str(image_path), verbose=False)[0]
    skeleton, conf = parse_skeleton(result)

    h, w = result.orig_shape if hasattr(result, "orig_shape") else (0, 0)

    return skeleton, conf, (w, h)


# ----------------------------
# LOAD CATEGORY JSON
# ----------------------------

def load_category(json_path: Path):
    with open(json_path, "r") as f:
        return json.load(f)


# ----------------------------
# BUILD DATASET
# ----------------------------

def build_dataset():
    model = load_model()

    all_samples = []

    json_files = [
        f for f in DATASET_DIR.glob("*.json")
        if f.name != OUTPUT_FILE.name
    ]

    print(f"[INFO] Found {len(json_files)} category files")

    for json_file in json_files:
        label_name = json_file.stem

        if label_name not in LABEL_MAP:
            print(f"[SKIP] {label_name}")
            continue

        label_id = LABEL_MAP[label_name]

        print(f"\n[LOAD] {label_name}")

        data = load_category(json_file)

        images = data.get("images", [])

        for img in tqdm(images, desc=f"Processing {label_name}"):

            image_path =  (DATASET_DIR / img["path"]).resolve()

            if not image_path.exists():
                print(f"[WARN] Missing image: {image_path}")
                continue

            try:
                skeleton, conf, frame_size = process_image(model, image_path)

                skeleton = normalize_skeleton(skeleton)

                all_samples.append({
                    "image_path": str(image_path),
                    "file_name": img["file_name"],
                    "label": label_id,
                    "label_name": label_name,
                    "skeleton": np.asarray(skeleton).tolist(),
                    "confidence": np.asarray(conf).tolist(),
                    "frame_size": list(frame_size),
                    "image_id": img["id"],
                })

            except Exception as e:
                print(f"[ERROR] {image_path}: {e}")

    print(f"\n[INFO] Total samples: {len(all_samples)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_samples, f)

    print(f"[SAVED] {OUTPUT_FILE}")


def normalize_skeleton(
    skeleton: list | np.ndarray,
    *,
    center_on_pelvis: bool = True,
    normalize_scale: bool = True,
    align_rotation: bool = False,
    min_scale: float = 1e-6,
) -> np.ndarray:
    normalizer = COCOSkeletonNormalizer(
        COCOSkeletonNormalizationConfig(
            center_on_pelvis=center_on_pelvis,
            normalize_scale=normalize_scale,
            align_rotation=align_rotation,
            min_scale=min_scale,
        )
    )
    return normalizer.normalize(skeleton).skeleton


# ----------------------------
# MAIN
# ----------------------------


def train_model():
    trainer = Trainer_class_COCOSkeletonTennis(
        json_path=OUTPUT_FILE,
        model_output_path=BASE_DIR / "models/skeleton_rf.joblib",
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    """Classe inutile mi serviva solo per costruire il json di training per COCO tennis"""
    train_model()
