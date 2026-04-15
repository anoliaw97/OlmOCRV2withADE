from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.dependencies import get_runtime
from backend.schemas import (
    MlDashboardResponse,
    MlDatasetBuildRequest,
    MlDatasetBuildResponse,
    MlTrainBatchRequest,
    MlTrainBatchResponse,
    MlPipelineRunRequest,
    MlPipelineRunResponse,
    MlPredictRequest,
    MlPredictResponse,
    MlTrainRequest,
    MlTrainResponse,
)


router = APIRouter(prefix="/api/ml", tags=["ml"])


@router.post("/dataset/build", response_model=MlDatasetBuildResponse)
def ml_build_dataset(request: MlDatasetBuildRequest) -> MlDatasetBuildResponse:
    runtime = get_runtime()
    if not runtime.packages:
        raise HTTPException(status_code=400, detail="No packages loaded. Load folder first.")

    output_csv = Path(request.output_csv).expanduser().resolve()
    try:
        rows, columns = runtime.ml_service.build_structured_dataset(runtime.packages, output_csv)
    except Exception as exc:
        runtime.log("error", f"ML dataset build failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to build structured dataset: {exc}") from exc

    runtime.log("status", f"ML dataset built: {rows} rows -> {output_csv}")
    return MlDatasetBuildResponse(
        ok=True,
        message=f"Structured dataset created with {rows} row(s).",
        output_csv=str(output_csv),
        rows=rows,
        columns=columns,
    )


@router.post("/train", response_model=MlTrainResponse)
def ml_train(request: MlTrainRequest) -> MlTrainResponse:
    runtime = get_runtime()
    dataset_csv = Path(request.dataset_csv).expanduser().resolve()

    try:
        artifact = runtime.ml_service.train_model(
            dataset_csv=dataset_csv,
            target=request.target,
            algorithm=request.algorithm,
        )
    except Exception as exc:
        runtime.log("error", f"ML train failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to train model: {exc}") from exc

    runtime.log("status", f"ML model trained for target={request.target}: {artifact.model_path}")
    return MlTrainResponse(
        ok=True,
        message="Model trained successfully.",
        model_path=str(artifact.model_path),
        target=artifact.target,
        algorithm=artifact.algorithm,
        metrics=artifact.metrics,
        feature_columns=artifact.feature_columns,
    )


@router.post("/train/batch", response_model=MlTrainBatchResponse)
def ml_train_batch(request: MlTrainBatchRequest) -> MlTrainBatchResponse:
    runtime = get_runtime()
    dataset_csv = Path(request.dataset_csv).expanduser().resolve()
    try:
        artifacts = runtime.ml_service.train_batch(
            dataset_csv=dataset_csv,
            targets=request.targets,
            algorithm=request.algorithm,
        )
    except Exception as exc:
        runtime.log("error", f"ML batch train failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed batch training: {exc}") from exc

    payload = [
        MlTrainResponse(
            ok=True,
            message="Model trained successfully.",
            model_path=str(item.model_path),
            target=item.target,
            algorithm=item.algorithm,
            metrics=item.metrics,
            feature_columns=item.feature_columns,
        )
        for item in artifacts
    ]

    runtime.log("status", f"ML batch training completed: {len(payload)} model(s).")
    return MlTrainBatchResponse(
        ok=True,
        message=f"Batch training completed: {len(payload)} model(s).",
        trained=payload,
    )


@router.post("/predict", response_model=MlPredictResponse)
def ml_predict(request: MlPredictRequest) -> MlPredictResponse:
    runtime = get_runtime()

    model_path = Path(request.model_path).expanduser().resolve() if request.model_path else None
    try:
        model_algo = "random_forest"
        if model_path is not None:
            lowered = model_path.name.lower()
            for candidate in ["lightgbm", "xgboost", "svr", "decision_tree", "linear", "ridge", "lasso", "elasticnet", "random_forest"]:
                if candidate in lowered:
                    model_algo = candidate
                    break
        prediction, used = runtime.ml_service.predict(
            target=request.target,
            features=request.features,
            model_path=model_path,
            algorithm=model_algo,
        )
    except Exception as exc:
        runtime.log("error", f"ML predict failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to predict: {exc}") from exc

    runtime.log("debug", f"ML prediction done for target={request.target}: {prediction:.6f}")
    return MlPredictResponse(
        ok=True,
        message="Prediction generated.",
        target=request.target,
        prediction=float(prediction),
        used_features={k: float(v) for k, v in used.items()},
    )


@router.get("/dashboard", response_model=MlDashboardResponse)
def ml_dashboard(dataset_csv: str = "data/ml/structured_dataset.csv", target: str = "Swc") -> MlDashboardResponse:
    runtime = get_runtime()
    path = Path(dataset_csv).expanduser().resolve()

    try:
        payload = runtime.ml_service.dashboard(path, target=target)
    except Exception as exc:
        runtime.log("error", f"ML dashboard failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to load dashboard data: {exc}") from exc

    return MlDashboardResponse(
        ok=True,
        message="Dashboard ready.",
        dataset_csv=payload["dataset_csv"],
        row_count=payload["row_count"],
        columns=payload["columns"],
        stats=payload["stats"],
        feature_importance=payload["feature_importance"],
        chart_points=payload["chart_points"],
        available_targets=payload["available_targets"],
    )


@router.post("/pipeline/run", response_model=MlPipelineRunResponse)
def ml_pipeline_run(request: MlPipelineRunRequest) -> MlPipelineRunResponse:
    runtime = get_runtime()
    if not runtime.packages:
        raise HTTPException(status_code=400, detail="No packages loaded. Load folder first.")

    try:
        steps = runtime.ml_service.run_pipeline_definition(
            packages=runtime.packages,
            pipeline_path=request.pipeline_path,
            pipeline_yaml_text=request.pipeline_yaml,
            default_target=request.default_target,
        )
    except Exception as exc:
        runtime.log("error", f"ML pipeline run failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to run ML pipeline: {exc}") from exc

    runtime.log("status", f"ML pipeline completed with {len(steps)} step(s).")
    return MlPipelineRunResponse(ok=True, message="ML pipeline run complete.", steps=steps)
