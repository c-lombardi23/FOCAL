"""
Configuration schema module for the Fiber Cleave Processing application.

This module defines Pydantic models for validating and loading JSON
configuration files for all CLI modes. Each mode has its own config class,
inheriting common fields and validators from BaseConfig, EarlyStoppingMixin,
and CheckpointMixin.
"""

import os
import json
from typing import List, Optional, Type, Dict
from pydantic import BaseModel, field_validator, model_validator


class EarlyStoppingMixin(BaseModel):
    """
    Adds early-stopping configuration parameters.

    Fields:
      - early_stopping (str): 'y' to enable early stopping, 'n' to disable.
      - patience (int): Number of epochs to wait for improvement.
      - monitor (str): Metric to monitor (e.g. 'val_loss', 'val_accuracy').
      - method (str): 'min' or 'max' to indicate direction of improvement.
    """

    early_stopping: Optional[str] = "n"
    patience: Optional[int] = 3
    monitor: Optional[str] = "val_accuracy"
    method: Optional[str] = "max"


class CheckpointMixin(BaseModel):
    """
    Adds model-checkpointing configuration parameters.

    Fields:
      - checkpoints (str): 'y' to enable checkpoints, 'n' to disable.
      - checkpoint_filepath (str): Path where to save the checkpoint file.
      - monitor (str): Metric to monitor for saving best model.
      - method (str): 'min' or 'max' depending on the monitored metric.
    """

    checkpoints: Optional[str] = "n"
    checkpoint_filepath: Optional[str] = None
    monitor: Optional[str] = "val_accuracy"
    method: Optional[str] = "max"


class BaseConfig(BaseModel):
    """
    Basic config for all classes.

    Fields:
      - csv_path: (str) : path to csv data for images
      - img_folder (str): folder containing images
      - mode: (str) : differnente modes e.g. test_cnn, train_cnn, etc.
      - set_mask (str): y/n for masking images to remove background
      - image_shape (list): array of image dimensions
      - feature_shape (list): array of feature dimensions
    """

    csv_path: str
    img_folder: str
    mode: str
    image_shape: List[int]
    set_mask: Optional[str] = None
    feature_shape: Optional[List[int]] = None

    @field_validator("csv_path", "img_folder", mode="before")
    @classmethod
    def path_exists(cls, value):
        if not os.path.exists(value):
            raise ValueError(f"{value} does not exist!")
        return value

    @field_validator("mode")
    @classmethod
    def valid_modes(cls, value):
        valid_modes = [
            "train_cnn",
            "train_mlp",
            "cnn_hyperparameter",
            "mlp_hyperparameter",
            "test_cnn",
            "test_mlp",
            "train_kfold_cnn",
            "train_kfold_mlp",
            "grad_cam",
            "train_image_only",
            "image_hyperparameter",
            "test_image_only",
            "custom_model",
            "train_xgboost",
            "test_xgboost",
        ]
        if value not in valid_modes:
            raise ValueError(f"{value} is not a valid mode!")
        return value


class ModelConfig(BaseConfig, EarlyStoppingMixin, CheckpointMixin):
    feature_scaler_path: Optional[str] = None
    label_scaler_path: Optional[str] = None
    model_path: Optional[str] = None
    learning_rate: Optional[float] = 0.001
    buffer_size: Optional[int] = 32
    batch_size: Optional[int] = 8
    test_size: Optional[float] = 0.2
    max_epochs: Optional[int] = None
    tuner_directory: Optional[str] = None
    objective: Optional[str] = None
    project_name: Optional[str] = None
    save_model_file: Optional[str] = None
    save_history_file: Optional[str] = None
    initial_epochs: Optional[int] = None
    continue_train: Optional[str] = None
    classification_path: Optional[str] = None
    encoder_path: Optional[str] = None


class TrainCNNConfig(ModelConfig):
    cnn_mode: str
    feature_shape: List[int]
    num_classes: int
    dropout1: float
    dense1: int
    dropout2: float
    dense2: int
    dropout3: float
    tension_threshold: Optional[int] = 190
    backbone: Optional[str] = "efficientnet"
    unfreeze_from: Optional[int] = None
    reduce_lr: Optional[float] = None
    reduce_lr_patience: Optional[int] = None
    classification_type: Optional[str] = ("binary",)
    backbone: Optional[str] = "mobilenet"

    @model_validator(mode="after")
    def valid_shapes(self):
        if self.feature_shape and self.feature_shape != [6]:
            raise ValueError("Feature shape must be 6 for CNN")
        if self.image_shape != [224, 224, 3]:
            raise ValueError("Image shape not compatible")
        return self


class TrainMLPConfig(ModelConfig):
    img_path: str

    @model_validator(mode="after")
    def valid_shapes(self):
        if self.feature_shape != [5]:
            raise ValueError("Feature shape must be 5 for MLP")
        if self.image_shape != [224, 224, 3]:
            raise ValueError("Image shape not compatible")
        return self


class TestCNNConfig(BaseConfig):
    cnn_mode: str
    tension_threshold: Optional[int] = 190
    tension_model_path: Optional[str] = None
    feature_scaler_path: Optional[str] = None
    model_path: Optional[str] = None
    test_features: Optional[List[float]] = None
    img_path: Optional[str] = None
    label_scaler_path: Optional[str] = None
    encoder_path: Optional[str] = None
    backbone: Optional[str] = None
    classification_path: Optional[str] = None

    @model_validator(mode="after")
    def valid_shapes(self):
        if self.feature_shape != [6]:
            raise ValueError("Feature shape must be 6 for CNN")
        if self.image_shape != [224, 224, 3]:
            raise ValueError("Image shape not compatible")
        return self


class TestMLPConfig(BaseConfig):
    feature_scaler_path: Optional[str] = None
    label_scaler_path: Optional[str] = None
    model_path: Optional[str] = None
    img_path: Optional[str] = None
    test_features: Optional[List[float]] = None

    @model_validator(mode="after")
    def valid_shapes(self):
        if self.feature_shape != [5]:
            raise ValueError("Feature shape must be 5 for MLP")
        if self.image_shape != [224, 224, 3]:
            raise ValueError("Image shape not compatible")
        return self


class TestImageOnlyConfig(BaseConfig):
    model_path: Optional[str] = None
    encoder_path: Optional[str] = None
    img_path: Optional[str] = None
    backbone: Optional[str] = None
    classification_type: Optional[str] = "binary"
    classification_path: Optional[str] = None


class TrainKFoldCNNConfig(TrainCNNConfig):
    pass


class TrainKFoldMLPConfig(TrainMLPConfig):
    pass


class TrainXGBoostConfig(ModelConfig):
    xgb_path: Optional[str] = None
    n_estimators: Optional[int] = 200
    max_depth: Optional[int] = 4
    random_state: Optional[int] = 42


class TestXGBoostConfig(TestMLPConfig):
    xgb_path: str


class GradCamConfig(BaseConfig):
    model_path: Optional[str] = None
    img_path: Optional[str] = None
    test_features: Optional[List[float]] = None
    backbone_name: Optional[str] = None
    conv_layer_name: Optional[str] = None
    heatmap_file: Optional[str] = None

    @model_validator(mode="after")
    def valid_shapes(self):
        if self.image_shape != [224, 224, 3]:
            raise ValueError("Image shape not compatible")
        return self


class TrainImageOnlyConfig(BaseConfig, EarlyStoppingMixin, CheckpointMixin):
    backbone: Optional[str] = "mobilenet"
    learning_rate: Optional[float] = 0.001
    buffer_size: Optional[int] = 32
    batch_size: Optional[int] = 8
    test_size: Optional[float] = 0.2
    dropout1_rate: Optional[float] = 0.1
    dense_units: Optional[int] = 32
    dropout2_rate: Optional[float] = 0.2
    l2_factor: Optional[float] = None
    max_epochs: Optional[int] = None
    tuner_directory: Optional[str] = None
    objective: Optional[str] = "val_accuracy"
    project_name: Optional[str] = None
    save_model_file: Optional[str] = None
    save_history_file: Optional[str] = None
    initial_epochs: Optional[int] = None
    continue_train: Optional[str] = None
    best_tuner_params: Optional[str] = None
    classification_path: Optional[str] = None
    num_classes: Optional[int] = 5
    reduce_lr: Optional[float] = None
    reduce_lr_patience: Optional[int] = None
    unfreeze_from: Optional[int] = None
    encoder_path: Optional[str] = None
    classification_type: Optional[str] = "binary"
    model_path: Optional[str] = None

    @model_validator(mode="after")
    def valid_shapes(self):
        if self.image_shape not in ([224, 224, 3], [224, 224, 1]):
            raise ValueError("Image shape not compatible")
        return self


class ImageHyperparameterConfig(TrainImageOnlyConfig):
    pass


MODE_TO_CONFIG: Dict[str, Type[BaseConfig]] = {
    "train_cnn": TrainCNNConfig,
    "train_mlp": TrainMLPConfig,
    "cnn_hyperparameter": TrainCNNConfig,
    "mlp_hyperparameter": TrainMLPConfig,
    "test_cnn": TestCNNConfig,
    "test_mlp": TestMLPConfig,
    "train_kfold_cnn": TrainKFoldCNNConfig,
    "train_kfold_mlp": TrainKFoldMLPConfig,
    "grad_cam": GradCamConfig,
    "train_image_only": TrainImageOnlyConfig,
    "test_image_only": TestImageOnlyConfig,
    "image_hyperparameter": ImageHyperparameterConfig,
    "custom_model": TrainImageOnlyConfig,
    "train_xgboost": TrainXGBoostConfig,
    "test_xgboost": TestXGBoostConfig
}


def load_config(filepath: str) -> BaseConfig:

    with open(filepath, "r") as f:
        data = json.load(f)
    mode = data.get("mode")
    config_cls = MODE_TO_CONFIG.get(mode)
    if config_cls is None:
        raise ValueError(f"Unknown or unimplemented mode: {mode}")
    return config_cls(**data)
