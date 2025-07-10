import mlflow
import mlflow.tensorflow


def log_cnn_training_run(config, model, history, artifacts=None):
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
        mlflow.log_params({
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
        })

        # Log metrics
        mlflow.log_metrics({
            "final_train_accuracy": history.history["accuracy"][-1],
            "final_val_accuracy": history.history["val_accuracy"][-1],
            "final_train_loss": history.history["loss"][-1],
            "final_val_loss": history.history["val_loss"][-1],
        })

        # Log model
        mlflow.tensorflow.log_model(model, "cnn_model")

        # Optional artifacts (e.g., saved model path or history CSV)
        if artifacts:
            for key, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=key)
