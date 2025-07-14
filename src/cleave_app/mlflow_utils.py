import os
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.tensorflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from tensorflow.keras.models import Model


def log_cnn_training_run(
    config,
    model: Model,
    history: Any,
    dataset_path: str,
    artifacts: Optional[Dict[str, Optional[str]]] = None,
) -> None:
    """
    Logs training run details for CNN models using MLflow.

    Args:
        config: Config object with hyperparameters.
        model: Trained TensorFlow model.
        history: Training history from model.fit().
        artifacts (dict): Optional dict with keys like "model", "history".
    """
    mlflow.set_experiment("fiber_cnn_training")
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "backbone": config.backbone,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "dropout1": config.dropout1,
                "dropout2": config.dropout2,
                "dropout3": config.dropout3,
                "dense1": config.dense1,
                "dense2": config.dense2,
                "brightness": config.brightness,
                "contrast": config.contrast,
                "rotation": config.rotation,
                "image_shape": config.image_shape,
                "feature_shape": config.feature_shape,
                "cnn_mode": config.cnn_mode,
            }
        )

        # Log metrics
        mlflow.log_metrics(
            {
                "final_train_accuracy": history.history["accuracy"][-1],
                "final_val_accuracy": history.history["val_accuracy"][-1],
                "final_train_loss": history.history["loss"][-1],
                "final_val_loss": history.history["val_loss"][-1],
            }
        )

        mlflow.log_artifact(dataset_path, artifact_path="dataset")
        df = pd.read_csv(dataset_path)
        mlflow.log_input(
            mlflow.data.from_pandas(
                df, source=dataset_path, name=str(dataset_path)
            )
        )
        # Log model
        mlflow.tensorflow.log_model(model, name="cnn_model")

        if artifacts:
            for key, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=key)


def log_mlp_training_run(
    config,
    model: Model,
    history: Any,
    dataset_path: str,
    artifacts: Optional[Dict[str, Optional[str]]] = None,
) -> None:
    """
    Logs training run details for MLP models using MLflow.

    Args:
        config: Config object with hyperparameters.
        model: Trained TensorFlow model.
        history: Training history from model.fit().
        artifacts (dict): Optional dict with keys like "model", "history".
    """
    mlflow.set_experiment("fiber_mlp_training")
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "backbone": config.backbone,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "dropout1": config.dropout1,
                "dropout2": config.dropout2,
                "dropout3": config.dropout3,
                "dense1": config.dense1,
                "dense2": config.dense2,
                "image_shape": config.image_shape,
                "feature_shape": config.feature_shape,
            }
        )

        # Log metrics
        mlflow.log_metrics(
            {
                "final_train_mae": history.history["mae"][-1],
                "final_val_mae": history.history["val_mae"][-1],
                "final_train_loss": history.history["loss"][-1],
                "final_val_loss": history.history["val_loss"][-1],
            }
        )
        mlflow.log_artifact(dataset_path, artifact_path="dataset")
        df = pd.read_csv(dataset_path)
        mlflow.log_input(
            mlflow.data.from_pandas(
                df, source=dataset_path, name=str(dataset_path)
            )
        )
        # Log model
        mlflow.tensorflow.log_model(model, name="mlp_model")

        if artifacts:
            for key, path in artifacts.items():
                if path is not None:
                    mlflow.log_artifact(path, artifact_path=key)
                else:
                    continue


def log_image_training_run(
    config,
    model: Model,
    history: Any,
    dataset_path: str,
    artifacts: Optional[Dict[str, Optional[str]]] = None,
) -> None:
    """
    Logs training run details for CNN models (image_only) using MLflow.

    Args:
        config: Config object with hyperparameters.
        model: Trained TensorFlow model.
        history: Training history from model.fit().
        artifacts (dict): Optional dict with keys like "model", "history".
    """
    mlflow.set_experiment("fiber_image_only_training")
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "backbone": config.backbone,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "dropout1": config.dropout1,
                "dense1": config.dense1,
                "dropout2": config.dropout2,
                "image_shape": config.image_shape,
                "feature_shape": config.feature_shape,
            }
        )

        # Log metrics
        mlflow.log_metrics(
            {
                "final_train_accuracy": history.history["accuracy"][-1],
                "final_val_accuracy": history.history["val_accuracy"][-1],
                "final_train_precision": history.history["precision"][-1],
                "final_val_precision": history.history["val_precision"][-1],
                "final_train_recall": history.history["recall"][-1],
                "final_val_recall": history.history["val_recall"][-1],
                "final_train_loss": history.history["loss"][-1],
                "final_val_loss": history.history["val_loss"][-1],
            }
        )

        # Log model
        mlflow.tensorflow.log_model(model, name="image_only_model")

        mlflow.log_artifact(dataset_path, artifact_path="dataset")
        df = pd.read_csv(dataset_path)
        mlflow.log_input(
            mlflow.data.from_pandas(
                df, source=dataset_path, name=str(dataset_path)
            )
        )

        if artifacts:
            for key, path in artifacts.items():
                if path is not None:
                    mlflow.log_artifact(path, artifact_path=key)
                else:
                    continue


def log_xgb_training_run(
    config,
    model: Model,
    X_train: Any,
    y_train: Any,
    dataset_path: str,
    evals_result: Optional[Dict[str, Optional[str]]] = None,
    artifacts: Optional[Dict[str, Optional[str]]] = None,
) -> None:
    """
    Logs training run details for CNN models using MLflow.

    Args:
        config: Config object with hyperparameters.
        model: Trained TensorFlow model.
        history: Training history from model.fit().
        artifacts (dict): Optional dict with keys like "model", "history".
    """
    mlflow.set_experiment("fiber_xbg_training")
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "objective": config.error_type,
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "random_state": config.random_state,
                "gamma": config.gamma,
                "subsample": config.subsample,
                "reg_lambda": config.reg_lambda,
            }
        )

        if config.error_type == "reg:absoluteerror":
            final_val_mae = evals_result["validation_1"]["mae"][-1]
            mlflow.log_metric("final_val_mae", final_val_mae)
        elif config.error_type == "reg:squarederror":
            final_val_rmse = evals_result["validation_1"]["rmse"][-1]
            mlflow.log_metric("final_val_rmse", final_val_rmse)
        else:
            raise ValueError("Incorrect error type!")

        signature = infer_signature(X_train, y_train)
        input_example = X_train[:2]
        input_example = pd.DataFrame(input_example)

        mlflow.log_artifact(dataset_path, artifact_path="dataset")
        df = pd.read_csv(dataset_path)
        mlflow.log_input(
            mlflow.data.from_pandas(
                df, source=dataset_path, name=str(dataset_path)
            )
        )
        # Log model
        mlflow.xgboost.log_model(
            model,
            name="xgb_model",
            signature=signature,
            input_example=input_example,
        )

        if artifacts:
            for key, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=key)


def log_classifier_test_results(
    tester,
    run_name: str,
    model_path: str,
    dataset_path: str,
    confusion_matrix_path: str,
    classification_path: str,
    roc_path: str,
    true_labels: List[int],
    pred_labels: List[int],
    predictions: List[float],
    image_only: Optional[bool] = False,
) -> None:

    if image_only:
        mlflow.set_experiment("image_only_results")
    else:
        mlflow.set_experiment("cnn_test_results")
    with mlflow.start_run(run_name=run_name):
        if os.path.exists(confusion_matrix_path):
            mlflow.log_artifact(confusion_matrix_path, artifact_path="plots")
        if os.path.exists(classification_path):
            mlflow.log_artifact(classification_path, artifact_path="plots")
        if os.path.exists(roc_path):
            mlflow.log_artifact(roc_path, artifact_path="plots")

        df = pd.DataFrame(
            {
                "true_labels": true_labels,
                "pred_labels": pred_labels,
                "predictions": [
                    (
                        round(float(p[0, 0]), 4)
                        if tester.classification_type == "binary"
                        else round(float(np.max(p[0])), 4)
                    )
                    for p in predictions
                ],
            }
        )
        model_dir = os.path.dirname(model_path)
        basename = os.path.basename(model_path)
        stem, _ = os.path.splitext(basename)
        if image_only:
            predictions_path = os.path.join(
            model_dir, f"{stem}_image_only_predictions.csv"
        )
        else:
            predictions_path = os.path.join(
                model_dir, f"{stem}_cnn_predictions.csv"
            )
        df.to_csv(predictions_path, index=False)
        mlflow.log_artifact(predictions_path, artifact_path="Predictions")

        mlflow.log_artifact(dataset_path, artifact_path="dataset")
        df = pd.read_csv(dataset_path)
        mlflow.log_input(
            mlflow.data.from_pandas(
                df, source=dataset_path, name=str(dataset_path)
            )
        )
