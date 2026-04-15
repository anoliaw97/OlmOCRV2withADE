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
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except Exception:
    lgb = None
    LIGHTGBM_AVAILABLE = False


NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")

TARGET_COLUMNS = [
    "Swc",
    "Sor",
    "Sgc",
    "Sgr",
    "Kro_max",
    "Krw_max",
    "Krg_max",
    "n_water",
    "m_oil",
    "n_gas",
]

SUPPLEMENTAL_TARGET_COLUMNS = [
    "Kklin_md",
    "phi",
    "RQI",
]

BASE_FEATURE_COLUMNS = [
    "Depth",
    "Kklin_md",
    "phi",
    "RQI",
    "Swc",
    "Sor",
    "Sgc",
    "Sgr",
    "Kro_max",
    "Krw_max",
    "Krg_max",
]

MODEL_HYPERPARAMS = {
    "random_forest": {
        "n_estimators": 250,
        "max_depth": 15,
        "min_samples_split": 5,
        "random_state": 42,
        "n_jobs": -1,
    },
    "decision_tree": {
        "max_depth": 10,
        "min_samples_leaf": 4,
        "random_state": 42,
    },
    "svr": {
        "kernel": "rbf",
        "C": 1.0,
        "epsilon": 0.08,
        "gamma": "scale",
    },
    "linear": {},
    "ridge": {"alpha": 1.0, "random_state": 42},
    "lasso": {"alpha": 0.01, "random_state": 42},
    "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5, "random_state": 42},
    "xgboost": {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.07,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    },
    "lightgbm": {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.07,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
}


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
            parsed = self._extract_scal_features(package)
            if parsed is not None:
                rows.append(parsed)

        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=["package_id", "base_name", *BASE_FEATURE_COLUMNS, *TARGET_COLUMNS])

        output = output_csv.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        return len(frame), list(frame.columns)

    def train_model(self, dataset_csv: Path, target: str, algorithm: str = "random_forest") -> TrainArtifact:
        csv_path = dataset_csv.expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        frame = pd.read_csv(csv_path)
        if frame.empty:
            raise ValueError("Dataset is empty. Build dataset first.")
        if target not in frame.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        train_frame = frame.copy()
        for col in BASE_FEATURE_COLUMNS + [target]:
            if col in train_frame.columns:
                train_frame[col] = pd.to_numeric(train_frame[col], errors="coerce")

        usable_features = [col for col in BASE_FEATURE_COLUMNS if col in train_frame.columns and col != target]
        if not usable_features:
            raise ValueError("No usable feature columns available.")

        train_frame = train_frame.dropna(subset=[target])
        if train_frame.empty:
            raise ValueError("No valid target values available.")

        for col in usable_features:
            if train_frame[col].isna().all():
                train_frame[col] = 0.0
            else:
                train_frame[col] = train_frame[col].fillna(float(train_frame[col].median()))

        X = train_frame[usable_features]
        y = train_frame[target]
        if len(train_frame) < 12:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        model, scaler = self._fit_model(algorithm, X_train, y_train)
        X_eval = scaler.transform(X_test) if scaler is not None else X_test
        y_pred = model.predict(X_eval)

        metrics = {
            "r2": float(r2_score(y_test, y_pred)) if len(y_test) > 1 else 0.0,
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }

        model_path = self._default_model_path(target, algorithm)
        payload = {
            "target": target,
            "algorithm": algorithm,
            "features": usable_features,
            "model": model,
            "scaler": scaler,
            "metrics": metrics,
        }
        with model_path.open("wb") as handle:
            pickle.dump(payload, handle)

        return TrainArtifact(
            model_path=model_path,
            target=target,
            algorithm=algorithm,
            metrics=metrics,
            feature_columns=usable_features,
        )

    def train_batch(self, dataset_csv: Path, targets: list[str], algorithm: str) -> list[TrainArtifact]:
        csv_path = dataset_csv.expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        selected = [t for t in targets if t]
        if not selected:
            frame = pd.read_csv(csv_path)
            selected = self._trainable_targets(frame)
        if not selected:
            raise ValueError("No trainable targets were found in dataset.")

        artifacts = []
        for target in selected:
            try:
                artifact = self.train_model(dataset_csv=dataset_csv, target=target, algorithm=algorithm)
                artifacts.append(artifact)
            except Exception:
                continue
        return artifacts

    def predict(
        self,
        target: str,
        features: dict[str, float],
        model_path: Path | None = None,
        algorithm: str = "random_forest",
    ) -> tuple[float, dict[str, float]]:
        path = model_path.expanduser().resolve() if model_path else self._default_model_path(target, algorithm)
        if not path.exists():
            candidates = sorted(self.ml_root.glob(f"model_{self._safe_target(target)}_*.pkl"))
            if not candidates:
                raise FileNotFoundError(f"Model not found: {path}")
            path = candidates[-1]

        with path.open("rb") as handle:
            payload = pickle.load(handle)

        model = payload["model"]
        scaler = payload.get("scaler")
        cols = list(payload.get("features", []))

        used = {}
        row = []
        for col in cols:
            value = float(features.get(col, 0.0))
            used[col] = value
            row.append(value)

        X = np.array([row], dtype=float)
        if scaler is not None:
            X = scaler.transform(X)
        pred = float(model.predict(X)[0])
        return pred, used

    def dashboard(self, dataset_csv: Path, target: str) -> dict[str, Any]:
        csv_path = dataset_csv.expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        frame = pd.read_csv(csv_path)

        stats: dict[str, dict[str, float]] = {}
        for col in [*BASE_FEATURE_COLUMNS, *TARGET_COLUMNS]:
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
        latest_model = self._latest_model_for_target(target)
        if latest_model is not None:
            with latest_model.open("rb") as handle:
                payload = pickle.load(handle)
            model = payload.get("model")
            cols = payload.get("features", [])
            if hasattr(model, "feature_importances_"):
                importance = {
                    str(col): float(score)
                    for col, score in zip(cols, model.feature_importances_, strict=False)
                }

        chart_points = []
        x_feature = "phi" if "phi" in frame.columns else "Kklin_md"
        if target in frame.columns and x_feature in frame.columns:
            for _, row in frame.head(500).iterrows():
                x = float(pd.to_numeric(row.get(x_feature, np.nan), errors="coerce") or np.nan)
                y = float(pd.to_numeric(row.get(target, np.nan), errors="coerce") or np.nan)
                if np.isnan(x) or np.isnan(y):
                    continue
                chart_points.append({"x": x, "y": y})

        available_targets = self._trainable_targets(frame)

        chosen_target = target if target in available_targets else (available_targets[0] if available_targets else target)
        if chosen_target != target:
            chart_points = []
            x_feature = "phi" if "phi" in frame.columns else "Kklin_md"
            if chosen_target in frame.columns and x_feature in frame.columns:
                for _, row in frame.head(500).iterrows():
                    x = float(pd.to_numeric(row.get(x_feature, np.nan), errors="coerce") or np.nan)
                    y = float(pd.to_numeric(row.get(chosen_target, np.nan), errors="coerce") or np.nan)
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
            "available_targets": available_targets,
            "selected_target": chosen_target,
        }

    def _trainable_targets(self, frame: pd.DataFrame) -> list[str]:
        preferred = [*TARGET_COLUMNS, *SUPPLEMENTAL_TARGET_COLUMNS]
        selected: list[str] = []
        for col in preferred:
            if col not in frame.columns:
                continue
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            if series.empty:
                continue
            selected.append(col)
        return selected

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
        algorithm = "random_forest"

        for component in components:
            if not isinstance(component, dict):
                continue
            name = str(component.get("name") or "unnamed-step")
            script = str(component.get("script") or "").lower()

            if "preprocess" in name.lower() or "preprocess" in script:
                out = self._component_output_path(component, "processed_data", dataset_csv)
                rows, cols = self.build_structured_dataset(packages, out)
                dataset_csv = out
                steps.append(f"{name}: dataset built rows={rows}, cols={len(cols)}")
                continue

            if "train" in name.lower() or "train" in script:
                target = self._component_target(component, target)
                algorithm = self._component_algorithm(component, algorithm)
                artifact = self.train_model(dataset_csv, target, algorithm)
                steps.append(
                    f"{name}: trained {artifact.algorithm} target={target} r2={artifact.metrics.get('r2', 0.0):.4f}"
                )
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

    def _extract_scal_features(self, package: DocumentPackage) -> dict[str, Any] | None:
        collected: dict[str, list[float]] = {
            "Depth": [],
            "Kklin_md": [],
            "phi": [],
            "RQI": [],
            "Swc": [],
            "Sor": [],
            "Sgc": [],
            "Sgr": [],
            "Kro_max": [],
            "Krw_max": [],
            "Krg_max": [],
            "n_water": [],
            "m_oil": [],
            "n_gas": [],
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
        for key in [*BASE_FEATURE_COLUMNS, *TARGET_COLUMNS]:
            vals = collected.get(key, [])
            row[key] = float(np.mean(vals)) if vals else np.nan

        if np.isnan(row.get("phi", np.nan)) and np.isnan(row.get("Kklin_md", np.nan)):
            return None

        if np.isnan(row.get("RQI", np.nan)) and not np.isnan(row.get("phi", np.nan)) and not np.isnan(
            row.get("Kklin_md", np.nan)
        ):
            phi = row["phi"]
            k = row["Kklin_md"]
            if phi > 0 and k > 0:
                row["RQI"] = float(0.0314 * np.sqrt(k / phi))

        return row

    def _scan_text_for_values(self, text: str, bucket: dict[str, list[float]]) -> None:
        for line in text.splitlines():
            lower = line.lower()
            numbers = [float(n) for n in NUMBER_PATTERN.findall(line)]
            if not numbers:
                continue

            if "depth" in lower:
                bucket["Depth"].extend(numbers)
            if "kklin" in lower or "kabs" in lower or "permeability" in lower:
                bucket["Kklin_md"].extend(numbers)
            if "porosity" in lower or "phi" in lower or "φ" in line:
                bucket["phi"].extend(numbers)
            if "rqi" in lower:
                bucket["RQI"].extend(numbers)
            if "swc" in lower:
                bucket["Swc"].extend(numbers)
            if "sor" in lower and "sorg" not in lower and "sorw" not in lower:
                bucket["Sor"].extend(numbers)
            if "sgc" in lower:
                bucket["Sgc"].extend(numbers)
            if "sgr" in lower:
                bucket["Sgr"].extend(numbers)
            if "kro" in lower and "max" in lower:
                bucket["Kro_max"].extend(numbers)
            if "krw" in lower and "max" in lower:
                bucket["Krw_max"].extend(numbers)
            if "krg" in lower and "max" in lower:
                bucket["Krg_max"].extend(numbers)
            if "n_water" in lower or "nw" in lower:
                bucket["n_water"].extend(numbers)
            if "m_oil" in lower or "mo" in lower:
                bucket["m_oil"].extend(numbers)
            if "n_gas" in lower or "ng" in lower:
                bucket["n_gas"].extend(numbers)

    def _fit_model(self, algorithm: str, X_train: pd.DataFrame, y_train: pd.Series):
        algo = (algorithm or "random_forest").strip().lower()
        scaler = None

        if algo == "linear":
            model = LinearRegression(**MODEL_HYPERPARAMS["linear"])
            model.fit(X_train, y_train)
            return model, scaler

        if algo == "ridge":
            model = Ridge(**MODEL_HYPERPARAMS["ridge"])
            model.fit(X_train, y_train)
            return model, scaler

        if algo == "lasso":
            model = Lasso(**MODEL_HYPERPARAMS["lasso"])
            model.fit(X_train, y_train)
            return model, scaler

        if algo == "elasticnet":
            model = ElasticNet(**MODEL_HYPERPARAMS["elasticnet"])
            model.fit(X_train, y_train)
            return model, scaler

        if algo == "decision_tree":
            model = DecisionTreeRegressor(**MODEL_HYPERPARAMS["decision_tree"])
            model.fit(X_train, y_train)
            return model, scaler

        if algo == "svr":
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_train)
            model = SVR(**MODEL_HYPERPARAMS["svr"])
            model.fit(Xs, y_train)
            return model, scaler

        if algo == "xgboost" and XGBOOST_AVAILABLE and xgb is not None:
            model = xgb.XGBRegressor(**MODEL_HYPERPARAMS["xgboost"])
            model.fit(X_train, y_train)
            return model, scaler

        if algo == "lightgbm" and LIGHTGBM_AVAILABLE and lgb is not None:
            model = lgb.LGBMRegressor(**MODEL_HYPERPARAMS["lightgbm"])
            model.fit(X_train, y_train)
            return model, scaler

        model = RandomForestRegressor(**MODEL_HYPERPARAMS["random_forest"])
        model.fit(X_train, y_train)
        return model, scaler

    def _default_model_path(self, target: str, algorithm: str) -> Path:
        safe_target = self._safe_target(target)
        safe_algo = re.sub(r"[^a-zA-Z0-9_\-]", "_", algorithm or "random_forest")
        return self.ml_root / f"model_{safe_target}_{safe_algo}.pkl"

    def _latest_model_for_target(self, target: str) -> Path | None:
        safe_target = self._safe_target(target)
        matches = sorted(self.ml_root.glob(f"model_{safe_target}_*.pkl"))
        if not matches:
            return None
        return matches[-1]

    def _safe_target(self, target: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", target)

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

    def _component_algorithm(self, component: dict[str, Any], default_algorithm: str) -> str:
        inputs = component.get("inputs")
        if isinstance(inputs, list):
            for item in inputs:
                if not isinstance(item, dict):
                    continue
                if "algorithm" in item:
                    return str(item["algorithm"]).strip() or default_algorithm
        if isinstance(inputs, dict) and "algorithm" in inputs:
            return str(inputs["algorithm"]).strip() or default_algorithm
        return default_algorithm

    def _read_metrics_for_target(self, target: str) -> dict[str, float]:
        latest = self._latest_model_for_target(target)
        if latest is None:
            return {}
        with latest.open("rb") as handle:
            payload = pickle.load(handle)
        metrics = payload.get("metrics", {})
        return {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (float, int, np.floating, np.integer))
        }
