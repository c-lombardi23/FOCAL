"""Hyperparameter tuning module for the Fiber Cleave Processing application.

This module provides classes for hyperparameter optimization using Keras
Tuner for both CNN and MLP models.
"""

import os
import warnings
from typing import Optional, Tuple

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

warnings.filterwarnings(
    "ignore", category=UserWarning, module="mlflow.data.tensorflow_dataset"
)


try:
    import tensorflow as tf
    from keras.applications import EfficientNetB0, MobileNetV2, ResNet50
    from keras_tuner import BayesianOptimization, Hyperband, HyperModel
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        BatchNormalization,
        Concatenate,
        Dense,
        Dropout,
        GlobalAveragePooling2D,
        Input,
        RandomBrightness,
        RandomContrast,
        RandomRotation,
        RandomZoom,
    )
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.regularizers import l2
except ImportError as e:
    print(f"Warning: Required ML libraries not found: {e}")
    print("Please install tensorflow>=2.19.0 and keras-tuner>=1.4.7")
    tf = None
    HyperModel = None
    Hyperband = None
    BayesianOptimization = None


class BuildHyperModel(HyperModel):
    """HyperModel for determining optimal hyperparameters for the combined
    CNN+MLP model.

    This class builds a model architecture that combines image features
    from MobileNetV2 with numerical parameters for fiber cleave
    classification.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        param_shape: Tuple[int, ...],
        backbone: Optional[str] = "mobilenet",
    ):
        """Initialize the hypermodel builder.

        Args:
            image_shape: Dimensions of input images (height, width, channels)
            param_shape: Dimensions of numerical parameters
        """
        if tf is None:
            raise ImportError("TensorFlow is required for BuildHyperModel")

        self.image_shape = image_shape
        self.param_shape = param_shape
        self.backbone = backbone

    def build(self, hp):
        """Build hypermodel to perform hyperparameter search.

        Args:
            hp: keras_tuner.engine.hyperparameters.HyperParameters
                Hyperparameters to be tuned

        Returns:
            tf.keras.Model: Compiled model with hyperparameters
        """
        # Pre-trained base model
        if self.backbone == "mobilenet":
            pre_trained_model = MobileNetV2(
                input_shape=self.image_shape,
                include_top=False,
                weights="imagenet",
                name="mobilenet",
            )
        elif self.backbone == "resnet":
            pre_trained_model = ResNet50(
                input_shape=self.image_shape,
                include_top=False,
                weights="imagenet",
                name="mobilenet",
            )
        elif self.backbone == "efficientnet":
            pre_trained_model = EfficientNetB0(
                input_shape=self.image_shape,
                include_top=False,
                weights="imagenet",
                name="efficientnetb0",
            )

        pre_trained_model.trainable = False

        # Data augmentation pipeline
        data_augmentation = Sequential(
            [
                RandomRotation(factor=hp.Float("rot", 0.0, 0.3, step=0.05)),
                RandomBrightness(
                    factor=hp.Float("bright", 0.0, 0.3, step=0.5)
                ),
                RandomZoom(
                    height_factor=hp.Float("height", 0.0, 0.2, step=0.5),
                    width_factor=hp.Float("width", 0.0, 0.2, step=0.5),
                ),
                RandomContrast(
                    factor=hp.Float("contrast", 0.0, 0.2, step=0.5)
                ),
            ]
        )

        # Image input and processing
        image_input = Input(shape=self.image_shape)
        x = data_augmentation(image_input)
        x = pre_trained_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float("dropout_1", 0.0, 0.3, step=0.1))(x)

        # Param input and processing
        param_input = Input(shape=self.param_shape)
        y = Dense(
            hp.Int("dense_1", min_value=32, max_value=64, step=16),
            activation="relu",
        )(param_input)
        y = Dropout(hp.Float("dropout_2", 0.0, 0.8, step=0.1))(y)
        y = BatchNormalization()(y)

        # Combine image and parameter features
        combined = Concatenate()([x, y])

        z = Dense(
            hp.Int("dense_2", min_value=32, max_value=64, step=16),
            activation="relu",
        )(combined)
        z = BatchNormalization()(z)
        z = Dropout(hp.Float("dropout_3", 0.0, 0.3, step=0.1))(z)
        z = Dense(1, activation="sigmoid")(z)

        model = Model(inputs=[image_input, param_input], outputs=z)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Choice(
                    "learning_rate",
                    values=[5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
                )
            ),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        return model


class HyperParameterTuning:
    """Class for tuning hyperparameters for the combined CNN+MLP model."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        feature_shape: Tuple[int, ...],
        max_epochs: int = 20,
        objective: str = "val_accuracy",
        directory: str = "./tuner_logs",
        project_name: str = "Cleave_Tuner",
        backbone: Optional[str] = "mobilenet",
        class_weights: Optional[str] = None,
    ) -> None:
        """Initialize the hyperparameter tuner.

        Args:
            image_shape: Dimensions of input images
            feature_shape: Dimensions of numerical features
            max_epochs: Maximum number of epochs to train for
            objective: Metric to monitor during tuning
            directory: Directory path to store hyperparameters
            project_name: Name of the tuning project
        """
        if (
            tf is None
            or HyperModel is None
            or Hyperband is None
            or BayesianOptimization is None
        ):
            raise ImportError(
                "TensorFlow and Keras Tuner are required for hyperparameter tuning"
            )

        self.image_shape = image_shape
        self.feature_shape = feature_shape
        self.backbone = backbone
        self.class_weights = class_weights
        hypermodel = BuildHyperModel(
            self.image_shape, self.feature_shape, self.backbone
        )
        self.tuner = BayesianOptimization(
            hypermodel,
            objective=objective,
            max_trials=max_epochs,
            num_initial_points=5,
            directory=directory,
            project_name=project_name,
        )
        """self.tuner = Hyperband( hypermodel, objective=objective,
        max_epochs=max_epochs, directory=directory,

        project_name=project_name )
        """

    def run_search(
        self,
        train_ds: "tf.data.Dataset",
        test_ds: "tf.data.Dataset",
        max_epochs: int,
    ) -> None:
        """Run hyperparameter search.

        Args:
            train_ds: tf.data.Dataset - Training dataset
            test_ds: tf.data.Dataset - Testing dataset
            max_epochs: maximum number of training epochs
        """
        es = EarlyStopping(monitor="val_loss", patience=5, mode="auto")
        self.tuner.search(
            train_ds,
            validation_data=test_ds,
            epochs=max_epochs,
            class_weight=self.class_weights,
            callbacks=[es],
        )

    def save_best_model(self, pathname: str) -> None:
        """Save the best model from hyperparameter search.

        Args:
            pathname: Path where to save the model
        """
        best_model = self.get_best_model()
        best_model.save(f"{pathname}")

    def get_best_model(self) -> "tf.keras.Model":
        """Get best model from hyperparameter search.

        Returns:
            tf.keras.Model: Best model from hyperparameter search
        """
        return self.tuner.get_best_models(num_models=1)[0]

    def get_best_hyperparameters(self) -> object:
        """Get best hyperparameters from hyperparameter search.

        Returns:
            keras_tuner.engine.hyperparameters.HyperParameters:
                Best hyperparameters from hyperparameter search
        """
        return self.tuner.get_best_hyperparameters(num_trials=1)[0]


class ImageOnlyHyperModel(HyperModel):
    """HyperModel for image-only classification (no numerical parameters)."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        num_classes: int = 5,
        backbone: Optional[str] = "mobilenet",
        classification_type: Optional[str] = "binary",
    ):
        """Initialize the image-only hypermodel.

        Args:
            image_shape: Dimensions of input images
            num_classes: Number of output classes
        """
        if tf is None:
            raise ImportError("TensorFlow is required for ImageOnlyHyperModel")

        self.image_shape = image_shape
        self.num_classes = num_classes
        self.backbone = backbone
        self.classification_type = classification_type

    def build(self, hp):
        """Build the image-only model with hyperparameters.

        Args:
            hp: Hyperparameters to tune

        Returns:
            tf.keras.Model: Compiled model
        """
        if self.backbone == "mobilenet":
            pre_trained_model = MobileNetV2(
                input_shape=self.image_shape,
                include_top=False,
                weights="imagenet",
                name="mobilenet",
            )
        elif self.backbone == "resnet":
            pre_trained_model = ResNet50(
                input_shape=self.image_shape,
                include_top=False,
                weights="imagenet",
                name="mobilenet",
            )
        elif self.backbone == "efficientnet":
            pre_trained_model = EfficientNetB0(
                input_shape=self.image_shape,
                include_top=False,
                weights="imagenet",
                name="efficientnetb0",
            )

        pre_trained_model.trainable = False

        # Data augmentation pipeline
        data_augmentation = Sequential(
            [
                RandomRotation(factor=hp.Float("rot", 0.0, 0.3, step=0.05)),
                RandomBrightness(
                    factor=hp.Float("bright", 0.0, 0.3, step=0.5)
                ),
                RandomZoom(
                    height_factor=hp.Float("height", 0.0, 0.2, step=0.5),
                    width_factor=hp.Float("width", 0.0, 0.2, step=0.5),
                ),
                RandomContrast(
                    factor=hp.Float("contrast", 0.0, 0.2, step=0.5)
                ),
            ]
        )

        image_input = Input(shape=self.image_shape)
        x = data_augmentation(image_input)
        x = pre_trained_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float("dropout", 0.0, 0.3, step=0.1))(x)

        l2_factor = hp.Choice("l2_factor", [0.001, 0.0001, 0.0])
        x = Dense(
            hp.Int("dense1", min_value=64, max_value=256, step=32),
            activation="relu",
            kernel_regularizer=l2(l2_factor),
        )(x)
        x = Dropout(hp.Float("dropout_2", 0.0, 0.4, step=0.1))(x)
        if self.classification_type == "binary":
            activation = "sigmoid"
            loss = "binary_crossentropy"
        elif self.classification_type == "multiclass":
            activation = "softmax"
            loss = "categorical_crossentropy"
        output = Dense(self.num_classes, activation=activation)(x)

        model = Model(inputs=image_input, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=hp.Choice(
                    "learning_rate", [5e-4, 1e-3, 5e-3, 0.01]
                )
            ),
            loss=loss,
            metrics=["accuracy"],
        )

        return model


class BuildMLPHyperModel(HyperModel):
    """HyperModel for MLP-based tension prediction.

    This class builds a model that uses features extracted from a pre-
    trained CNN to predict optimal tension values.
    """

    def __init__(self, model_path: str):
        """Initialize the MLP hypermodel.

        Args:
            model_path: Path to the pre-trained CNN model
        """
        if tf is None:
            raise ImportError("TensorFlow is required for BuildMLPHyperModel")

        self.cnn_model = tf.keras.models.load_model(model_path)
        self.image_input = self.cnn_model.input[0]
        self.feature_output = self.cnn_model.get_layer("global_avg").output

    def build(self, hp):
        """Build model with hyperparameters.

        Args:
            hp: keras_tuner.HyperParameters - Hyperparameters to be used for tuning

        Returns:
            tf.keras.Model: Model to be trained
        """
        x = Dropout(hp.Float("dropout_1", 0.0, 0.4, step=0.1))(
            self.feature_output
        )
        feature_input = Input(shape=(4,), name="feature_input")

        y = Dense(
            hp.Int("dense_1", min_value=16, max_value=64, step=16),
            activation="relu",
        )(feature_input)
        y = Dropout(hp.Float("dropout_2", 0.0, 0.8, step=0.2))(y)

        combined = Concatenate()([x, y])

        z = Dense(
            hp.Int("dense_2", min_value=16, max_value=64, step=16),
            activation="relu",
        )(combined)
        z = Dropout(hp.Float("dropout_3", 0.0, 0.4, step=0.1))(z)
        z = Dense(1)(z)

        mlp_hypermodel = Model(
            inputs=[self.image_input, feature_input], outputs=z
        )

        mlp_hypermodel.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Choice(
                    "learning_rate", values=[0.0005, 0.001, 0.005, 0.01]
                )
            ),
            loss="mse",
            metrics=["mae"],
        )

        return mlp_hypermodel


class ImageHyperparameterTuning(HyperParameterTuning):
    """Hyperparameter tuning specifically for image-only models."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        max_epochs: int = 20,
        objective: str = "val_accuracy",
        directory: str = "./tuner_logs",
        project_name: str = "CNN_Image_Only",
        backbone: Optional[str] = "mobilenet",
        num_classes: Optional[int] = 5,
        class_weights: Optional[str] = None,
    ):
        """Initialize image-only hyperparameter tuning.

        Args:
            image_shape: Dimensions of input images
            max_epochs: Maximum number of epochs
            objective: Metric to optimize
            directory: Directory for tuner logs
            project_name: Name of the tuning project
        """
        self.class_weights = class_weights
        self.image_shape = image_shape
        hypermodel = ImageOnlyHyperModel(
            self.image_shape,
            num_classes=num_classes,
            backbone=backbone,
        )

        self.tuner = BayesianOptimization(
            hypermodel,
            objective=objective,
            max_trials=max_epochs,
            num_initial_points=5,
            directory=directory,
            project_name=project_name,
        )
        """self.tuner = Hyperband( hypermodel, objective=objective,
        max_epochs=max_epochs, directory=directory,

        project_name=project_name )
        """


class MLPHyperparameterTuning(HyperParameterTuning):
    """Hyperparameter tuning specifically for MLP models."""

    def __init__(
        self,
        cnn_path: str,
        max_epochs: int = 20,
        objective: str = "val_mae",
        directory: str = "./tuner_logs",
        project_name: str = "MLPTuner",
        class_weights=None,
    ):
        """Initialize MLP hyperparameter tuning.

        Args:
            cnn_path: Path to the pre-trained CNN model
            max_epochs: Maximum number of epochs
            objective: Metric to optimize
            directory: Directory for tuner logs
            project_name: Name of the tuning project
        """
        self.cnn_model = tf.keras.models.load_model(cnn_path)
        self.image_input = self.cnn_model.input[0]
        self.feature_output = self.cnn_model.get_layer("global_avg").output
        self.class_weights = class_weights
        hypermodel = BuildMLPHyperModel(cnn_path)
        self.tuner = Hyperband(
            hypermodel,
            objective=objective,
            max_epochs=max_epochs,
            directory=directory,
            project_name=project_name,
        )
