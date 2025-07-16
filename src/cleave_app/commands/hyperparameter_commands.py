"""This module defines the logic for optimizing hyperparameters for each model."""

import mlflow.keras

from cleave_app.data_processing import DataCollector, MLPDataCollector
from cleave_app.hyperparameter_tuning import (HyperParameterTuning,
                                              ImageHyperparameterTuning,
                                              MLPHyperparameterTuning)

from .base_command import BaseCommand
from .utils import _run_search_helper


class CNNHyperparameterSearch(BaseCommand):
    """Perform hyperparameter search for CNN model."""

    def _execute_command(self, config) -> None:
        mlflow.set_experiment(config.project_name)
        mlflow.keras.autolog()

        data = DataCollector(
            config.csv_path,
            config.img_folder,
            backbone=config.backbone,
            angle_threshold=config.angle_threshold,
            diameter_threshold=config.diameter_threshold,
        )
        images, features, labels = data.extract_data()
        train_ds, test_ds, class_weights = data.create_datasets(
            images,
            features,
            labels,
            config.test_size,
            config.buffer_size,
            config.batch_size,
            feature_scaler_path=config.feature_scaler_path,
        )

        max_epochs = config.max_epochs or 20

        tuner = HyperParameterTuning(
            config.image_shape,
            config.feature_shape,
            max_epochs=max_epochs,
            project_name=config.project_name,
            directory=config.tuner_directory,
            backbone=config.backbone,
            class_weights=class_weights,
        )

        _run_search_helper(config, tuner, train_ds, test_ds)


class MLPHyperparameterSearch(BaseCommand):
    """Perform hyperparameter search for MLP model."""

    def _execute_command(self, config) -> None:
        mlflow.set_experiment
        mlflow.keras.autolog(config.project_name)

        data = MLPDataCollector(
            config.csv_path,
            config.img_folder,
            angle_threshold=config.angle_threshold,
            diameter_threshold=config.diameter_threshold,
        )
        images, features, labels = data.extract_data()
        train_ds, test_ds = data.create_datasets(
            images,
            features,
            labels,
            config.test_size,
            config.buffer_size,
            config.batch_size,
            feature_scaler_path=None,
            tension_scaler_path=None,
        )

        max_epochs = config.max_epochs or 20
        tuner = MLPHyperparameterTuning(
            config.model_path,
            max_epochs=max_epochs,
            project_name=config.project_name,
            directory=config.tuner_directory,
        )

        _run_search_helper(config, tuner, train_ds, test_ds)


class ImageHyperparameterSearch(BaseCommand):
    """Perform hyperparameter search for image-only model."""

    def execute(self, config) -> None:
        mlflow.set_experiment
        mlflow.keras.autolog(config.project_name)

        data = DataCollector(
            config.csv_path,
            config.img_folder,
            backbone=config.backbone,
            set_mask=config.set_mask,
            classification_type=config.classification_type,
            angle_threshold=config.angle_threshold,
            diameter_threshold=config.diameter_threshold,
        )
        images, features, labels = data.extract_data()
        train_ds, test_ds, class_weights = data.create_datasets(
            images,
            features,
            labels,
            config.test_size,
            config.buffer_size,
            config.batch_size,
        )

        # Convert to image-only datasets
        train_ds = data.image_only_dataset(train_ds)
        test_ds = data.image_only_dataset(test_ds)

        max_epochs = config.max_epochs or 20
        tuner = ImageHyperparameterTuning(
            config.image_shape,
            max_epochs=max_epochs,
            project_name=config.project_name,
            directory=config.tuner_directory,
            backbone=config.backbone,
            num_classes=config.num_classes,
            class_weights=class_weights,
        )

        _run_search_helper(
            config,
            tuner,
            train_ds,
            test_ds,
            best_params_path=config.best_tuner_params,
        )
