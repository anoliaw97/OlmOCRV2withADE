from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.loaders import DocumentPackage

try:
    import yaml
except Exception:
    yaml = None

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")

FEATURE_COLUMNS = [
    "porosity_mean",
    "permeability_mean",
    "swirr_mean",
    "kro_end_mean",
    "krw_end_mean",
    "pc_entry_mean",
]


@dataclass(slots=True)
class TrainArtifact:
    model_path: Path
    target: str
    algorithm: str
    metrics: dict[str, float]
    feature_columns: list[str]


class MlPipelineService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.ml_root = project_root / "data" / "ml"
        self.ml_root.mkdir(parents=True, exist_ok=True)

    def build_structured_dataset(self, packages: list[DocumentPackage], output_csv: Path) -> tuple[int, list[str]]:
        rows = []
        for package in packages:
            parsed = self._extract_numeric_features(package)
            if parsed:
                rows.append(parsed)

        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=["package_id", "base_name", *FEATURE_COLUMNS])

        output = output_csv.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        return len(frame), list(frame.columns)

    def train_model(self, dataset_csv: Path, target: str) -> TrainArtifact:
        csv_path = dataset_csv.expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        frame = pd.read_csv(csv_path)
        if frame.empty:
            raise ValueError("Dataset is empty. Build dataset first.")
        if target not in frame.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        usable_features = [col for col in FEATURE_COLUMNS if col in frame.columns and col != target]
        if not usable_features:
            raise ValueError("No usable feature columns found for training.")

        train_frame = frame.copy()
        for col in usable_features + [target]:
            train_frame[col] = pd.to_numeric(train_frame[col], errors="coerce")
        train_frame = train_frame.dropna(subset=[target])
        if train_frame.empty:
            raise ValueError("No valid target values available after cleaning.")

        for col in usable_features:
            if train_frame[col].isna().all():
                train_frame[col] = 0.0
            else:
                train_frame[col] = train_frame[col].fillna(float(train_frame[col].median()))

        X = train_frame[usable_features]
        y = train_frame[target]
        if len(train_frame) < 6:
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X, y)
            y_pred = model.predict(X)
            metrics = {
                "r2": float(r2_score(y, y_pred)) if len(train_frame) > 1 else 0.0,
                "mae": float(mean_absolute_error(y, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            }
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            model = RandomForestRegressor(n_estimators=300, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                "r2": float(r2_score(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            }

        model_path = self._default_model_path(target)
        payload = {
            "target": target,
            "algorithm": "RandomForestRegressor",
            "features": usable_features,
            "model": model,
            "metrics": metrics,
        }
        with model_path.open("wb") as handle:
            pickle.dump(payload, handle)

        return TrainArtifact(
            model_path=model_path,
            target=target,
            algorithm="RandomForestRegressor",
            metrics=metrics,
            feature_columns=usable_features,
        )

    def predict(self, target: str, features: dict[str, float], model_path: Path | None = None) -> tuple[float, dict[str, float]]:
        path = model_path.expanduser().resolve() if model_path else self._default_model_path(target)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Train model first.")

        with path.open("rb") as handle:
            payload = pickle.load(handle)

        model = payload["model"]
        cols = list(payload.get("features", []))
        used = {}
        row = []
        for col in cols:
            value = float(features.get(col, 0.0))
            used[col] = value
            row.append(value)

        pred = float(model.predict([row])[0])
        return pred, used

    def dashboard(self, dataset_csv: Path, target: str) -> dict[str, Any]:
        csv_path = dataset_csv.expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        frame = pd.read_csv(csv_path)

        stats: dict[str, dict[str, float]] = {}
        for col in FEATURE_COLUMNS:
            if col not in frame.columns:
                continue
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            if series.empty:
                continue
            stats[col] = {
                "mean": float(series.mean()),
                "min": float(series.min()),
                "max": float(series.max()),
                "std": float(series.std(ddof=0)),
            }

        importance = {}
        model_path = self._default_model_path(target)
        if model_path.exists():
            with model_path.open("rb") as handle:
                payload = pickle.load(handle)
            model = payload.get("model")
            cols = payload.get("features", [])
            if hasattr(model, "feature_importances_"):
                importance = {
                    str(col): float(score)
                    for col, score in zip(cols, model.feature_importances_, strict=False)
                }

        chart_points = []
        if target in frame.columns:
            for _, row in frame.head(300).iterrows():
                x = float(pd.to_numeric(row.get("porosity_mean", np.nan), errors="coerce") or np.nan)
                y = float(pd.to_numeric(row.get(target, np.nan), errors="coerce") or np.nan)
                if np.isnan(x) or np.isnan(y):
                    continue
                chart_points.append({"x": x, "y": y})

        return {
            "dataset_csv": str(csv_path),
            "row_count": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
            "stats": stats,
            "feature_importance": importance,
            "chart_points": chart_points,
        }

    def run_pipeline_definition(
        self,
        packages: list[DocumentPackage],
        pipeline_path: str,
        pipeline_yaml_text: str,
        default_target: str,
    ) -> list[str]:
        definition = self._load_pipeline_definition(pipeline_path, pipeline_yaml_text)
        components = definition.get("components", []) if isinstance(definition, dict) else []
        if not isinstance(components, list) or not components:
            raise ValueError("Pipeline definition has no components.")

        steps = []
        dataset_csv = self.ml_root / "structured_dataset.csv"
        target = default_target

        for component in components:
            if not isinstance(component, dict):
                continue
            name = str(component.get("name") or "unnamed-step")
            script = str(component.get("script") or "").lower()

            if "preprocess" in name.lower() or "preprocess" in script:
                out = self._component_output_path(component, "processed_data", dataset_csv)
                rows, cols = self.build_structured_dataset(packages, out)
                dataset_csv = out
                steps.append(f"{name}: built dataset rows={rows}, cols={len(cols)}")
                continue

            if "train" in name.lower() or "train" in script:
                target = self._component_target(component, target)
                artifact = self.train_model(dataset_csv, target)
                steps.append(f"{name}: trained {artifact.algorithm} for target={target}")
                continue

            if "evaluate" in name.lower() or "evaluate" in script:
                dash = self.dashboard(dataset_csv, target)
                metrics = self._read_metrics_for_target(target)
                steps.append(
                    f"{name}: evaluated target={target}, rows={dash['row_count']}, r2={metrics.get('r2', 0.0):.4f}"
                )
                continue

            steps.append(f"{name}: skipped (no mapped action)")

        return steps

    def _extract_numeric_features(self, package: DocumentPackage) -> dict[str, Any] | None:
        collected = {
            "porosity_mean": [],
            "permeability_mean": [],
            "swirr_mean": [],
            "kro_end_mean": [],
            "krw_end_mean": [],
            "pc_entry_mean": [],
        }

        for path in package.extracted_paths():
            if not path.exists() or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            self._scan_text_for_values(text, collected)

        row: dict[str, Any] = {
            "package_id": package.package_id,
            "base_name": package.base_name,
        }

        for key in FEATURE_COLUMNS:
            values = collected.get(key, [])
            row[key] = float(np.mean(values)) if values else np.nan

        if all(np.isnan(row[col]) for col in FEATURE_COLUMNS):
            return None
        return row

    def _scan_text_for_values(self, text: str, bucket: dict[str, list[float]]) -> None:
        for line in text.splitlines():
            lower = line.lower()
            numbers = [float(n) for n in NUMBER_PATTERN.findall(line)]
            if not numbers:
                continue

            if "porosity" in lower:
                bucket["porosity_mean"].extend(numbers)
            if "perm" in lower:
                bucket["permeability_mean"].extend(numbers)
            if "swirr" in lower or "irreducible water" in lower:
                bucket["swirr_mean"].extend(numbers)
            if "kro" in lower:
                bucket["kro_end_mean"].extend(numbers)
            if "krw" in lower:
                bucket["krw_end_mean"].extend(numbers)
            if "pc" in lower and ("entry" in lower or "capillary" in lower):
                bucket["pc_entry_mean"].extend(numbers)

    def _default_model_path(self, target: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", target)
        return self.ml_root / f"model_{safe}.pkl"

    def _load_pipeline_definition(self, pipeline_path: str, pipeline_yaml_text: str) -> dict[str, Any]:
        if pipeline_yaml_text.strip():
            if yaml is None:
                raise RuntimeError("PyYAML is not installed. Install requirements and retry.")
            payload = yaml.safe_load(pipeline_yaml_text)
            return payload if isinstance(payload, dict) else {}

        if not pipeline_path.strip():
            raise ValueError("Provide pipeline_path or pipeline_yaml.")
        path = Path(pipeline_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")

        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is not installed. Install requirements and retry.")
            payload = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
            return payload if isinstance(payload, dict) else {}

        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return payload if isinstance(payload, dict) else {}

    def _component_output_path(self, component: dict[str, Any], key: str, default: Path) -> Path:
        outputs = component.get("outputs")
        if isinstance(outputs, list):
            for item in outputs:
                if not isinstance(item, dict):
                    continue
                if key in item:
                    return Path(str(item[key])).expanduser().resolve()
        if isinstance(outputs, dict) and key in outputs:
            return Path(str(outputs[key])).expanduser().resolve()
        return default

    def _component_target(self, component: dict[str, Any], default_target: str) -> str:
        inputs = component.get("inputs")
        if isinstance(inputs, list):
            for item in inputs:
                if not isinstance(item, dict):
                    continue
                if "target" in item:
                    return str(item["target"]).strip() or default_target
        if isinstance(inputs, dict) and "target" in inputs:
            return str(inputs["target"]).strip() or default_target
        return default_target

    def _read_metrics_for_target(self, target: str) -> dict[str, float]:
        path = self._default_model_path(target)
        if not path.exists():
            return {}
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        metrics = payload.get("metrics", {})
        return {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (float, int, np.floating, np.integer))
        }
