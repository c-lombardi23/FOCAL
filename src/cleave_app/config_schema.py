"""Configuration schema module for the Fiber Cleave Processing application.

This module defines Pydantic models for validating and loading JSON
configuration files for all CLI modes. Each mode has its own config
class, inheriting common fields and validators from BaseConfig,
EarlyStoppingMixin, and CheckpointMixin.
"""

# Import necessary standard library modules for JSON handling and OS path checks.
import json
import os
from pathlib import Path
# Import typing hints for better code clarity and static analysis.
from typing import Dict, List, Literal, Optional, Type

# Import core components from Pydantic for data validation and modeling.
from pydantic import BaseModel, field_validator, model_validator


# Define a Mixin class for Early Stopping parameters.
class EarlyStoppingMixin(BaseModel):
    """Adds early-stopping configuration parameters."""

    # Flag to enable ('y') or disable ('n') the early stopping callback.
    early_stopping: Optional[str] = "n"
    # Number of epochs with no improvement after which training will be stopped.
    patience: Optional[int] = 3
    # The metric to be monitored for early stopping (e.g., 'val_accuracy').
    monitor: Optional[str] = "val_accuracy"
    # The direction of improvement ('min' for loss, 'max' for accuracy).
    method: Literal["max", "min"] = "max"


# Define a Mixin class for Model Checkpointing parameters.
class CheckpointMixin(BaseModel):
    """Adds model-checkpointing configuration parameters."""

    # Flag to enable ('y') or disable ('n') the model checkpoint callback.
    checkpoints: Optional[str] = "n"
    # Filepath where the best model checkpoint will be saved.
    checkpoint_filepath: Optional[Path] = None
    # The metric to be monitored for saving the best model.
    monitor: Optional[str] = "val_accuracy"
    # The direction of improvement for the monitored metric.
    method: Literal["max", "min"] = "max"


# Define the base configuration class that all other configs inherit from.
class BaseConfig(BaseModel):
    """Basic config for all classes."""

    # Path to the CSV file containing metadata for the images.
    csv_path: Path
    # Path to the directory containing all the image files.
    img_folder: Path
    # The operational mode for the CLI (e.g., 'train_cnn', 'test_mlp').
    mode: str
    # The target shape for images after resizing (height, width, channels).
    image_shape: List[int]
    # Flag to enable ('y') or disable ('n') background masking on images.
    set_mask: Optional[str] = None
    # The shape of the numerical feature vector.
    feature_shape: Optional[List[int]] = None

    # Define a Pydantic validator for the 'csv_path' and 'img_folder' fields.
    @field_validator("csv_path", "img_folder", mode="before")
    @classmethod
    def path_exists(cls, value):
        # Check if the provided file or directory path actually exists.
        if not os.path.exists(value):
            # If the path does not exist, raise a ValueError.
            raise ValueError(f"{value} does not exist!")
        # If the path exists, return the value to be used.
        return value

    # Define a Pydantic validator for the 'mode' field.
    @field_validator("mode")
    @classmethod
    def valid_modes(cls, value):
        # Check if the provided mode is in the list of valid modes.
        if value not in MODE_TO_CONFIG.keys():
            # If the mode is not valid, raise a ValueError.
            raise ValueError(f"{value} is not a valid mode!")
        # If the mode is valid, return the value.
        return value


# Define a comprehensive configuration class for training models.
class ModelConfig(BaseConfig, EarlyStoppingMixin, CheckpointMixin):
    """
    Configuration for building, training, and evaluating a machine learning model.

    This class centralizes all hyperparameters and paths needed for the entire
    ML pipeline, from data loading and augmentation to model training, tuning,
    and saving artifacts. It inherits settings for early stopping and model
    checkpointing from its mixins.
    """

    # --- Data Augmentation Parameters ---
    # Maximum brightness adjustment factor for data augmentation.
    brightness: Optional[float] = 0.0
    # Maximum rotation angle in degrees for data augmentation.
    rotation: Optional[float] = 0.0
    # Maximum vertical shift as a fraction of image height for augmentation.
    height: Optional[float] = 0.0
    # Maximum horizontal shift as a fraction of image width for augmentation.
    width: Optional[float] = 0.0
    # Maximum contrast adjustment factor for data augmentation.
    contrast: Optional[float] = 0.0

    # --- File Path Configurations ---
    # Path to save or load the feature scaler (e.g., StandardScaler).
    feature_scaler_path: Optional[str] = None
    # Path to save or load the label scaler (for regression tasks).
    label_scaler_path: Optional[str] = None
    # Path to load a pre-trained CNN model or save the current one.
    model_path: Optional[str] = None
    # Path to save the final classification report
    classification_path: Optional[str] = None
    # Path to save or load a One-Hot Encoder.
    encoder_path: Optional[str] = None
    # Path to save the trained model file.
    save_model_file: Optional[str] = None
    # Path to save the training history
    save_history_file: Optional[str] = None

    # --- Hyperparameter Tuning ---
    # Directory to store Keras Tuner trial history and results.
    tuner_directory: Optional[str] = None
    # Name of the tuning project for Keras Tuner.
    project_name: Optional[str] = None

    # --- Training Hyperparameters ---
    # Initial learning rate for the optimizer.
    learning_rate: Optional[float] = 0.001
    # Size of the shuffle buffer for the tf.data.Dataset pipeline.
    buffer_size: Optional[int] = 32
    # Number of samples per batch to use during model training.
    batch_size: Optional[int] = 8
    # Fraction of the dataset to be reserved for the test set.
    test_size: Optional[float] = 0.2
    # Maximum number of epochs to train the model.
    max_epochs: Optional[int] = None
    # The objective metric for callbacks to monitor (e.g., 'val_loss').
    objective: Optional[str] = None
    # The epoch number to start or continue training from.
    initial_epochs: Optional[int] = None
    # Flag ('y'/'n') to indicate if training should resume from a checkpoint.
    continue_train: Optional[str] = None


# Define configuration specific to training the hybrid CNN+MLP model.
class TrainCNNConfig(ModelConfig):
    # Specifies the mode within CNN training.
    cnn_mode: str
    # The shape of the numerical feature vector.
    feature_shape: List[int]
    # The number of output classes for the classification layer.
    num_classes: int
    # Dropout rate for the first dropout layer in the CNN head.
    dropout1: float
    # Number of neurons in the first dense layer of the numerical branch.
    dense1: int
    # Dropout rate for the second dropout layer.
    dropout2: float
    # Number of neurons in the second dense layer of the classification head.
    dense2: int
    # Dropout rate for the final dropout layer before the output.
    dropout3: float
    # The angle threshold used for binary classification.
    angle_threshold: float
    # The diameter threshold used for binary classification.
    diameter_threshold: float
    # The tension threshold used for binary classification.
    tension_threshold: Optional[int] = 190
    # The name of the pre-trained backbone to use (e.g., 'efficientnet').
    backbone: Optional[str] = "efficientnet"
    # Layer index from which to start unfreezing weights for fine-tuning.
    unfreeze_from: Optional[int] = None
    # Factor by which to reduce learning rate on a plateau.
    reduce_lr: Optional[float] = None
    # Number of epochs to wait before reducing the learning rate.
    reduce_lr_patience: Optional[int] = None
    # The type of classification ('binary' or 'multiclass').
    classification_type: Literal["binary", "multiclass"] = "binary"

    # Define a Pydantic model validator that runs after initial field validation.
    @model_validator(mode="after")
    def valid_shapes(self):
        # Validate that the feature shape is correct for this mode.
        if self.feature_shape and self.feature_shape != [5]:
            raise ValueError("Feature shape must be 5 for CNN")
        # Validate that the image shape is compatible with standard models.
        if self.image_shape != [224, 224, 3]:
            raise ValueError("Image shape not compatible")
        # Return the validated model instance.
        return self


# Define configuration specific to training the MLP-only model.
class TrainMLPConfig(ModelConfig):
    # backbone name from CNN training
    backbone: str
    # Dropout rate for the first dropout layer in the CNN head.
    dropout1: float
    # Number of neurons in the first dense layer of the numerical branch.
    dense1: int
    # Dropout rate for the second dropout layer.
    dropout2: float
    # Number of neurons in the second dense layer of the classification head.
    dense2: int
    # Dropout rate for the final dropout layer before the output.
    dropout3: float
    # Path to the image (used for context, not for training the MLP).
    img_path: Optional[str] = None
    num_classes: int
    angle_threshold: float
    diameter_threshold: float
    reduce_lr: Optional[float] = None
    # Number of epochs to wait before reducing the learning rate.
    reduce_lr_patience: Optional[int] = None

    # Define a post-validation check for this model.
    @model_validator(mode="after")
    def valid_shapes(self):
        # Validate the feature shape required for the MLP model.
        if self.feature_shape != [4]:
            raise ValueError("Feature shape must be 4 for MLP")
        # Validate the image shape (for consistency, though not used in training).
        if self.image_shape != [224, 224, 3]:
            raise ValueError("Image shape not compatible")
        # Return the validated model instance.
        return self


# Define configuration specific to testing the hybrid CNN+MLP model.
class TestCNNConfig(BaseConfig):
    classification_threshold: Optional[float] = 0.5
    # Specifies the mode within CNN testing.
    cnn_mode: str
    # The tension threshold for classification logic.
    tension_threshold: Optional[int] = 190
    # Path to a trained tension prediction model.
    tension_model_path: Optional[str] = None
    # Path to a saved feature scaler.
    feature_scaler_path: Optional[str] = None
    # Path to the trained CNN model file.
    model_path: Optional[str] = None
    # A list of numerical features to use for a single test prediction.
    test_features: Optional[List[float]] = None
    # Path to a single image for testing.
    img_path: Optional[str] = None
    # Path to a saved label scaler.
    label_scaler_path: Optional[str] = None
    # Path to a saved label encoder.
    encoder_path: Optional[str] = None
    # The name of the pre-trained backbone used in the model.
    backbone: Optional[str] = None
    # Path to save the output classification report.
    classification_path: Optional[str] = None
    angle_threshold: float
    diameter_threshold: float

    # Define a post-validation check for this model.
    @model_validator(mode="after")
    def valid_shapes(self):
        # Validate the required feature shape for CNN testing.
        if self.feature_shape != [5]:
            raise ValueError("Feature shape must be 5 for CNN")
        # Validate the required image shape for CNN testing.
        if self.image_shape != [224, 224, 3]:
            raise ValueError("Image shape not compatible")
        # Return the validated model instance.
        return self


# Define configuration specific to testing the MLP-only model.
class TestMLPConfig(BaseConfig):
    angle_threshold: float
    diameter_threshold: float
    # Path to a saved label scaler.
    label_scaler_path: Optional[str] = None
    # Path to the trained MLP model file.
    model_path: Optional[str] = None
    # Path to a single image for context.
    img_path: Optional[str] = None
    # A list of numerical features for a single test prediction.
    test_features: Optional[List[float]] = None

    # Define a post-validation check for this model.
    @model_validator(mode="after")
    def valid_shapes(self):
        # Validate the required feature shape for MLP testing.
        if self.feature_shape != [4]:
            raise ValueError("Feature shape must be 4 for MLP")
        # Validate the image shape for consistency.
        if self.image_shape != [224, 224, 3]:
            raise ValueError("Image shape not compatible")
        # Return the validated model instance.
        return self


# Define configuration for testing an image-only classification model.
class TestImageOnlyConfig(BaseConfig):
    angle_threshold: float
    diameter_threshold: float
    # Path to the trained image-only model file.
    model_path: Optional[str] = None
    # Path to the saved label encoder.
    encoder_path: Optional[str] = None
    # Path to a single image for testing.
    img_path: Optional[str] = None
    # The name of the backbone used in the model.
    backbone: Optional[str] = None
    # The type of classification ('binary' or 'multiclass').
    classification_type: Literal["binary", "multiclass"] = "binary"
    # Path to save the output classification report.
    classification_path: Optional[str] = None


# Define a config for K-Fold Cross-Validation on the CNN model.
class TrainKFoldCNNConfig(TrainCNNConfig):
    # This class inherits all fields and validators from TrainCNNConfig.
    pass


# Define a config for K-Fold Cross-Validation on the MLP model.
class TrainKFoldMLPConfig(TrainMLPConfig):
    # This class inherits all fields and validators from TrainMLPConfig.
    pass


# Define configuration for training an XGBoost model.
class TrainXGBoostConfig(ModelConfig):
    # MSA or RMSE
    error_type: str
    # Path to save the trained XGBoost model.
    xgb_path: Optional[str] = None
    # Number of boosting rounds (trees) in the XGBoost model.
    n_estimators: Optional[int] = 200
    # Maximum depth of each tree in the XGBoost model.
    max_depth: Optional[int] = 4
    # Seed for the random number generator for reproducibility.
    random_state: Optional[int] = 42
    # Minimum loss reduction
    gamma: Optional[float] = 0.0
    # Subsample ratio to prevent overfitting
    subsample: Optional[float] = 1.0
    # L2 regularization
    reg_lambda: Optional[float] = 1.0
    angle_threshold: float
    diameter_threshold: float


# Define configuration for testing an XGBoost model.
class TestXGBoostConfig(TestMLPConfig):
    # Path to the trained XGBoost model file.
    xgb_path: str
    # The angle threshold for classification logic.
    angle_threshold: float
    # The diameter threshold for classification logic.
    diameter_threshold: float


# Define configuration for generating Grad-CAM heatmaps.
class GradCamConfig(BaseModel):
    mode: str
    image_shape: List[int]
    # Path to the trained model for which to generate the heatmap.
    model_path: Optional[str] = None
    # Path to the input image for Grad-CAM.
    img_path: Optional[str] = None
    # Numerical features required if the model is multi-input.
    test_features: Optional[List[float]] = None
    # The name of the backbone model (for finding layers).
    backbone: Optional[str] = None
    # The name of the final convolutional layer to visualize.
    conv_layer_name: Optional[str] = None
    # Filepath to save the output heatmap image.
    heatmap_file: Optional[str] = None

    # Define a post-validation check for this model.
    @model_validator(mode="after")
    def valid_shapes(self):
        # Validate that the image shape is compatible.
        if self.image_shape != [224, 224, 3]:
            raise ValueError("Image shape not compatible")
        # Return the validated model instance.
        return self


# Define configuration for training an image-only classification model.
class TrainImageOnlyConfig(ModelConfig):
    """Configuration for training an image-only classification model."""

    backbone: str
    unfreeze_from: Optional[int] = 0
    # Max angle for good cleave
    angle_threshold: float
    # Max diameter for good cleave
    diameter_threshold: float
    # Dropout 1 rate
    dropout1: Optional[float] = 0.1
    # Number of FC layers
    dense1: Optional[int] = 32
    # Dropout before final layer
    dropout2: Optional[float] = 0.2
    # L2 regularization
    l2_factor: Optional[float] = None
    # Tuner params from best search
    best_tuner_params: Optional[str] = None
    # Number of classes to use
    num_classes: Optional[int] = 5
    classification_type: str
    reduce_lr: Optional[float] = None
    # Number of epochs to wait before reducing the learning rate.
    reduce_lr_patience: Optional[int] = None

    # --- Data Augmentation Parameters ---
    # Maximum brightness adjustment factor for data augmentation.
    brightness: Optional[float] = 0.0
    # Maximum rotation angle in degrees for data augmentation.
    rotation: Optional[float] = 0.0
    # Maximum vertical shift as a fraction of image height for augmentation.
    height: Optional[float] = 0.0
    # Maximum horizontal shift as a fraction of image width for augmentation.
    width: Optional[float] = 0.0
    # Maximum contrast adjustment factor for data augmentation.
    contrast: Optional[float] = 0.0

    # Define a post-validation check for this model.
    @model_validator(mode="after")
    def valid_shapes(self):
        # Allow both 3-channel (RGB) and 1-channel (grayscale) images.
        if self.image_shape not in ([224, 224, 3], [224, 224, 1]):
            raise ValueError("Image shape not compatible")
        # Return the validated model instance.
        return self


class CNNHyperparameterConfig(BaseConfig):
    tuner_directory: str
    project_name: str
    model_path: str
    backbone: str
    angle_threshold: float
    diameter_threshold: float
    test_size: float
    batch_size: int
    buffer_size: int
    max_epochs: int
    save_model_file: Optional[str]


# Define a config for hyperparameter tuning the image-only model.
class ImageHyperparameterConfig(TrainImageOnlyConfig):
    # This class inherits all fields from TrainImageOnlyConfig.
    pass


class MLPHyperparameterConfig(BaseConfig):
    tuner_directory: str
    project_name: str
    model_path: str
    backbone: str
    angle_threshold: float
    diameter_threshold: float
    test_size: float
    batch_size: int
    buffer_size: int
    max_epochs: int
    save_model_file: Optional[str]


class TrainRLConfig(BaseModel):
    csv_path: str
    cnn_path: str
    agent_path: str
    mode: str
    img_folder: str
    threshold: float
    max_steps: Optional[int] = 15
    feature_shape: List[int]
    buffer_size: Optional[int] = 1000000
    batch_size: Optional[int] = 256
    tau: Optional[float] = 0.1
    learning_rate: Optional[float] = 0.0001
    timesteps: Optional[int] = 5000
    low_range: Optional[float] = 0.7
    high_range: Optional[float] = 1.4
    max_delta: Optional[float] = 5.0


class TestRLConfig(TrainRLConfig):
    episodes: int
    run_name: str


# A dictionary mapping the 'mode' string to its corresponding Pydantic config class.
MODE_TO_CONFIG: Dict[str, Type[BaseConfig]] = {
    "train_cnn": TrainCNNConfig,
    "train_mlp": TrainMLPConfig,
    "cnn_hyperparameter": CNNHyperparameterConfig,
    "mlp_hyperparameter": MLPHyperparameterConfig,
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
    "test_xgboost": TestXGBoostConfig,
    "train_rl": TrainRLConfig,
    "test_rl": TestRLConfig,
}


# A factory function to load the correct configuration class based on the mode.
def load_config(filepath: str) -> BaseConfig:
    """Loads a configuration object from a JSON file based on the 'mode' field."""

    # Open the specified JSON file for reading.
    with open(filepath, "r") as f:
        # Load the JSON data into a Python dictionary.
        data = json.load(f)
    # Get the 'mode' value from the loaded data.
    mode = data.get("mode")
    # Look up the corresponding config class in the mapping dictionary.
    config_cls = MODE_TO_CONFIG.get(mode)
    # Check if a valid class was found for the mode.
    if config_cls is None:
        # If no class is found, raise an error.
        raise ValueError(f"Unknown or unimplemented mode: {mode}")
    # Instantiate the correct config class with the loaded data and return it.
    return config_cls(**data)
