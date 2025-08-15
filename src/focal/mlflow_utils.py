import os
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.tensorflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature
from tensorflow.keras.models import Model


def log_cnn_training_run(
    config: dict,
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
        best_epoch = np.argmax(history.history["val_accuracy"])
        # Log metrics
        mlflow.log_metrics(
            {
                "best_train_accuracy": history.history["accuracy"][best_epoch],
                "best_val_accuracy": history.history["val_accuracy"][
                    best_epoch
                ],
                "best_train_precision": history.history["precision"][
                    best_epoch
                ],
                "best_val_precision": history.history["val_precision"][
                    best_epoch
                ],
                "best_train_recall": history.history["recall"][best_epoch],
                "best_val_recall": history.history["val_recall"][best_epoch],
                "best_train_loss": history.history["loss"][best_epoch],
                "best_val_loss": history.history["val_loss"][best_epoch],
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
        mlflow.tensorflow.log_model(
            model,
            name="cnn_model",
        )

        if artifacts:
            for key, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=key)


def log_mlp_training_run(
    config: dict,
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

        best_epoch = np.argmin(history.history["val_mae"])
        # Log metrics
        mlflow.log_metrics(
            {
                "best_train_mae": history.history["mae"][best_epoch],
                "best_val_mae": history.history["val_mae"][best_epoch],
                "best_train_loss": history.history["loss"][best_epoch],
                "best_val_loss": history.history["val_loss"][best_epoch],
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
    config: dict,
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
        dataset_path: Path to dataset used during training.
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

        best_epoch = np.argmax(history.history["val_accuracy"])

        # Log metrics
        mlflow.log_metrics(
            {
                "best_train_accuracy": history.history["accuracy"][best_epoch],
                "best_val_accuracy": history.history["val_accuracy"][
                    best_epoch
                ],
                "best_train_precision": history.history["precision"][
                    best_epoch
                ],
                "best_val_precision": history.history["val_precision"][
                    best_epoch
                ],
                "best_train_recall": history.history["recall"][best_epoch],
                "best_val_recall": history.history["val_recall"][best_epoch],
                "best_train_loss": history.history["loss"][best_epoch],
                "best_val_loss": history.history["val_loss"][best_epoch],
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
    config: dict,
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
        X_train: Train features of dataset.
        y_train: Train labels of dataset.
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
    tester: Any,
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
    """Logs summary to mlflow for testing cnn.

    Args:
        tester (Any): TestCNN class instance
        run_name (str): name of the mlflow run
        model_path (str): path to the cnn model
        dataset_path (str): path to the csv dataset
        confusion_matrix_path (str): path to save confusion matrix
        classification_path (str): path to save classification report
        roc_path (str): path to save roc plot
        true_labels (List[int]): true prediction labels
        pred_labels (List[int]): prediction results
        predictions (List[float]): float values of predictions
        image_only (Optional[bool], optional): if you want to test on only images
    """

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


def log_regressor_test_results(
    model_path: str,
    run_name: str,
    dataset_path: str,
    experiment_name: str,
    tensions: List[float],
    predicted_delta: List[float],
    predictions: List[float],
    true_delta: List[float],
    mean: float,
) -> None:
    """Log mlp or xgb regression results to mlflow.

    Args:
        model_path (str): path to the trained model
        run_name (str): name of mlflow run
        dataset_path (str): path to the csv dataset
        experiment_name (str): name of mlflow experiment
        tensions (List[float]): list of current tensions
        predicted_delta (List[float]): list of predicted change in tensions
        predictions (List[float]): list of predicted absolute tensions
        true_delta (List[float]): list of true delta in tension

    Raises:
        Exception: experiment failed to be created
    """

    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except MlflowException as e:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise Exception(
                f"Failed to get or create experiment '{experiment_name}'"
            )

        experiment_id = experiment.experiment_id
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        mlflow.log_artifact(dataset_path, artifact_path="dataset")
        df_ds = pd.read_csv(dataset_path)
        mlflow.log_input(
            mlflow.data.from_pandas(
                df_ds, source=dataset_path, name=str(dataset_path)
            )
        )
        df = pd.DataFrame(
            {
                "current_tension": tensions,
                "true_delta": true_delta,
                "pred_delta": predicted_delta,
                "pred_t": predictions,
            }
        )
        df = df.round(4)
        model_dir = os.path.dirname(model_path)
        basename = os.path.basename(model_path)
        stem, _ = os.path.splitext(basename)

        predictions_path = os.path.join(model_dir, f"{stem}_performance.csv")
        df.to_csv(predictions_path, index=False)
        mlflow.log_artifact(predictions_path, artifact_path="Predictions")


def log_cnn_hyperparameter(
    config: dict,
    best_hp: dict,
    run_name: str,
    experiment_name: Optional[str] = "cnn_hyperparameter",
) -> None:
    """Logs cnn hyperparameter search to mlflow.

    Args:
        config (json file): configuration file with parameters
        best_hp (Dict): best hyperparameters from search
        run_name (str): name of mlflow run
        experiment_name (str, optional): Defaults to "cnn_hyperparameter".
    """

    if not mlflow.set_experiment(experiment_name):
        mlflow.create_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "batch_size": config.batch_size,
                "test_size": config.test_size,
                "angle_threshold": config.angle_threshold,
                "diameter_threshold": config.angle_threshold,
            }
        )
        mlflow.log_params(best_hp)


def log_rl_test(
    config: dict,
    run_name: str,
    info: List[Dict[str, Any]],
    experiment_name: Optional[str] = "rl_testing",
) -> None:
    """Logs reinforcement learning test to mlflow.

    Args:
        config (json file): configuration file with parameters
        run_name (str): name of mlflow run
        info (Dict): dictionary of saved info from training rl
        experiment_name (str, optional): Defaults to "rl_testing".
    """
    if not mlflow.set_experiment(experiment_name):
        mlflow.create_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "buffer_size": config.buffer_size,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "max_delta": config.max_delta,
                "high_range": config.high_range,
                "low_range": config.low_range,
                "max_steps": config.max_steps,
                "threshold": config.threshold,
                "tau": config.tau,
                "timesteps": config.timesteps,
            }
        )

        mlflow.log_dict({"episodes": info}, "test_episodes.json")


def log_mlp_hyperparameter(
    config: dict,
    best_hp: dict,
    run_name: str,
    experiment_name: Optional[str] = "mlp_hyperparameter",
) -> None:
    """Logs mlp hyperparamter results to mlflow.

    Args:
        config (Any): json config file
        best_hp (Any): best hyperparameters from search
        run_name (str): name of mlflow run
        experiment_name (Optional[str], optional): Defaults to "mlp_hyperparameter".
    """

    if not mlflow.set_experiment(experiment_name):
        mlflow.create_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "batch_size": config.batch_size,
                "test_size": config.test_size,
                "angle_threshold": config.angle_threshold,
                "diameter_threshold": config.angle_threshold,
            }
        )
        mlflow.log_params(best_hp)
