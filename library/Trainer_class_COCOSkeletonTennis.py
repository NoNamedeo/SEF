from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


class Trainer_class_COCOSkeletonTennis:
    """
    Trainer per classificazione movimenti da skeleton COCO (17 keypoints).
    """

    def __init__(
        self,
        json_path: str | Path,
        model_output_path: str | Path = "skeleton_model.joblib",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.json_path = Path(json_path)
        self.model_output_path = Path(model_output_path)
        self.test_size = test_size
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=self.random_state,
        )

    # ----------------------------
    # LOAD DATASET
    # ----------------------------
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        with open(self.json_path, "r") as f:
            data = json.load(f)

        X, y = [], []

        for sample in data:
            skeleton = np.array(sample["skeleton"], dtype=np.float32).flatten()
            label = sample["label"]

            X.append(skeleton)
            y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        print(f"[DATA] Loaded samples: {len(X)}")
        print(f"[DATA] Feature shape: {X.shape}")

        return X, y

    # ----------------------------
    # TRAIN
    # ----------------------------
    def train(self):
        X, y = self.load_dataset()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        print("[TRAIN] Training model...")

        self.model.fit(X_train, y_train)

        print("[TRAIN] Evaluating...")

        preds = self.model.predict(X_test)

        print("\n[REPORT]")
        print(classification_report(y_test, preds))

        return self.model

    # ----------------------------
    # SAVE MODEL
    # ----------------------------
    def save_model(self):
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, self.model_output_path)

        print(f"[SAVE] Model saved to: {self.model_output_path}")

    # ----------------------------
    # LOAD MODEL
    # ----------------------------
    def load_model(self):
        if not self.model_output_path.exists():
            raise FileNotFoundError(self.model_output_path)

        self.model = joblib.load(self.model_output_path)

        print(f"[LOAD] Model loaded from: {self.model_output_path}")

        return self.model

    # ----------------------------
    # PREDICT SINGLE SAMPLE
    # ----------------------------
    def predict(self, skeleton: np.ndarray) -> int:
        """
        skeleton: shape (17,2) oppure (34,)
        """
        if skeleton.ndim == 2:
            skeleton = skeleton.flatten()

        skeleton = np.asarray(skeleton, dtype=np.float32).reshape(1, -1)

        return int(self.model.predict(skeleton)[0])