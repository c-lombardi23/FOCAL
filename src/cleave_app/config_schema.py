from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Type, Dict
import os

class EarlyStoppingMixin(BaseModel):
    early_stopping: Optional[str] = "n"
    patience: Optional[int] = 3
    monitor: Optional[str] = "val_accuracy"
    method: Optional[str] = "max"

class CheckpointMixin(BaseModel):
    checkpoints: Optional[str] = "n"
    checkpoint_filepath: Optional[str] = None
    monitor: Optional[str] = "val_accuracy"
    method: Optional[str] = "max"

class BaseConfig(BaseModel):
    csv_path: str
    img_folder: str
    mode: str
    image_shape: List[int]
    set_mask: Optional[str]
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
            'train_cnn', 'train_mlp',
            'cnn_hyperparameter', 'mlp_hyperparameter',
            'test_cnn', 'test_mlp', 'train_kfold_cnn', 'train_kfold_mlp',
            'grad_cam', 'train_image_only', 'image_hyperparameter', 'test_image_only', 'custom_model'
        ]
        if value not in valid_modes:
            raise ValueError(f"{value} is not a valid mode!")
        return value

class TrainCNNConfig(BaseConfig, EarlyStoppingMixin, CheckpointMixin):
    feature_shape: List[int]
    backbone: Optional[str] = "mobilenet"
    feature_scaler_path: Optional[str] = None
    label_scaler_path: Optional[str] = None
    model_path: Optional[str] = None
    learning_rate: Optional[float] = 0.001
    buffer_size: Optional[int] = 32
    batch_size: Optional[int] = 8
    test_size: Optional[float] = 0.2
    max_epochs: Optional[int] = None
    tuner_directory: Optional[str] = None
    objective: Optional[str] = "val_accuracy"
    project_name: Optional[str] = None
    save_model_file: Optional[str] = None
    save_history_file: Optional[str] = None
    unfreeze_from: Optional[int] = None
    reduce_lr: Optional[float] = None
    reduce_lr_patience: Optional[int] = None
    initial_epochs: Optional[int] = None
    continue_train: Optional[str] = None
    classification_path: Optional[str] = None
    encoder_path: Optional[str] = None
    num_classes: Optional[int] = 5

    @model_validator(mode="after")
    def valid_shapes(self):
        if self.feature_shape:
            if self.feature_shape != [6]:
                raise ValueError("Feature shape must be 6 for CNN")
        if self.image_shape != [224, 224, 3]:
            raise ValueError("Image shape not compatible")
        return self

class TrainMLPConfig(BaseConfig, EarlyStoppingMixin, CheckpointMixin):
    feature_scaler_path: Optional[str] = None
    label_scaler_path: Optional[str] = None
    model_path: Optional[str] = None
    learning_rate: Optional[float] = 0.001
    buffer_size: Optional[int] = 32
    batch_size: Optional[int] = 8
    test_size: Optional[float] = 0.2
    max_epochs: Optional[int] = None
    tuner_directory: Optional[str] = None
    objective: Optional[str] = "val_mae"
    project_name: Optional[str] = None
    save_model_file: Optional[str] = None
    save_history_file: Optional[str] = None
    initial_epochs: Optional[int] = None
    continue_train: Optional[str] = None
    classification_path: Optional[str] = None

    @model_validator(mode="after")
    def valid_shapes(self):
        if self.feature_shape != [5]:
            raise ValueError("Feature shape must be 5 for MLP")
        if self.image_shape != [224, 224, 3]:
            raise ValueError("Image shape not compatible")
        return self

class CNNHyperparameterConfig(TrainCNNConfig):
    pass

class MLPHyperparameterConfig(TrainMLPConfig):
    pass

class TestCNNConfig(BaseConfig):
    feature_scaler_path: Optional[str] = None
    model_path: Optional[str] = None
    test_features: Optional[List[float]] = None
    img_path: Optional[str] = None
    label_scaler_path: Optional[str] = None,
    encoder_path: Optional[str] = None
    backbone: Optional[str] = None

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
    encoder_path: str  
    img_path: Optional[str] = None
    backbone: Optional[str] = None

class TrainKFoldCNNConfig(TrainCNNConfig):
    pass

class TrainKFoldMLPConfig(TrainMLPConfig):
    pass

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
    l2_factor: Optional[float] =None
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
    reduce_lr_patience: Optional[int] = None,
    unfreeze_from: Optional[int] = None,
    encoder_path: Optional[str] = None

    @model_validator(mode="after")
    def valid_shapes(self):
        if self.image_shape != [224, 224, 3] and self.image_shape != [224, 224, 1]:
            raise ValueError("Image shape not compatible")
        return self

class ImageHyperparameterConfig(TrainImageOnlyConfig):
    pass

MODE_TO_CONFIG: Dict[str, Type[BaseConfig]] = {
    'train_cnn': TrainCNNConfig,
    'train_mlp': TrainMLPConfig,
    'cnn_hyperparameter': CNNHyperparameterConfig,
    'mlp_hyperparameter': MLPHyperparameterConfig,
    'test_cnn': TestCNNConfig,
    'test_mlp': TestMLPConfig,
    'train_kfold_cnn': TrainKFoldCNNConfig,
    'train_kfold_mlp': TrainKFoldMLPConfig,
    'grad_cam': GradCamConfig,
    'train_image_only': TrainImageOnlyConfig,
    'test_image_only':TestImageOnlyConfig,
    'image_hyperparameter': ImageHyperparameterConfig,
    'custom_model': TrainImageOnlyConfig
}

def load_config(filepath: str) -> BaseConfig:
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
    mode = data.get("mode")
    config_cls = MODE_TO_CONFIG.get(mode)
    if config_cls is None:
        raise ValueError(f"Unknown or unimplemented mode: {mode}")
    return config_cls(**data)

    
    








