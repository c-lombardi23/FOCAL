"""
Model pipeline module for the Fiber Cleave Processing application.

This module provides classes for building, training, and managing CNN and MLP models
for fiber cleave quality classification and tension prediction.
"""

import os
import warnings
from typing import Optional, Tuple, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras.layers import (
        RandomRotation,
        RandomBrightness,
        RandomZoom,
        GaussianNoise,
        RandomContrast,
        Dense,
        Concatenate,
        Input,
        Dropout,
        BatchNormalization,
        GlobalAveragePooling2D,
        Conv2D,
        MaxPooling2D,
        Activation,
    )
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.applications import (
        MobileNetV2,
        ResNet50,
        EfficientNetB0,
    )
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
        TensorBoard,
    )
except ImportError as e:
    print(f"Warning: TensorFlow not found: {e}")
    print("Please install tensorflow>=2.19.0")
    tf = None


class CustomModel:
    """
    Class for defining custom models using pre-trained MobileNetV2.

    This class provides functionality for building, compiling, and training
    CNN models for fiber cleave classification.
    """

    def __init__(
        self,
        train_ds: "tf.data.Dataset",
        test_ds: "tf.data.Dataset",
        num_classes: int,
        classification_type: Optional[str] = "binary",
    ) -> None:
        """
        Initialize the custom model.

        Args:
            train_ds: Training dataset
            test_ds: Test dataset
            num_classes: Number of output classes
            classification_type: Type of classification
        """
        if tf is None:
            raise ImportError("TensorFlow is required for CustomModel")

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.classification_type = classification_type
        self.num_classes = num_classes

    def _get_backbone_model(
        self, backbone: str, image_shape: Tuple[int, int, int]
    ) -> tf.keras.Model:
        """
        Get pretrained backbone model based on specified backbone type.

        Args:
            backbone: Type of backbone model ("mobilenet", "resnet", "efficientnet")
            image_shape: Input image shape (height, width, channels)

        Returns:
            tf.keras.Model: Pretrained backbone model

        Raises:
            ValueError: If backbone type is not supported
        """
        if backbone == "mobilenet":
            pre_trained_model = MobileNetV2(
                input_shape=image_shape,
                include_top=False,
                weights="imagenet",
                name="mobilenet",
            )
        elif backbone == "resnet":
            pre_trained_model = ResNet50(
                input_shape=image_shape,
                include_top=False,
                weights="imagenet",
                name="resnet50",
            )
        elif backbone == "efficientnet":
            pre_trained_model = EfficientNetB0(
                input_shape=image_shape,
                include_top=False,
                weights="imagenet",
                name="efficientnetb0",
            )
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. Supported backbones: mobilenet, resnet, efficientnet"
            )

        return pre_trained_model

    def _build_custom_model(
        self, image_shape: Tuple[int, int, int], num_classes: int = 5
    ) -> "tf.keras.Model":
        data_augmentation = Sequential(
            [
                RandomRotation(factor=0.02),
                RandomBrightness(factor=0.02),
            ]
        )

        image_input = Input(shape=image_shape)
        x = data_augmentation(image_input)

        x = Conv2D(16, (5, 5), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(4, 4))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = GlobalAveragePooling2D()(x)

        x = Dense(16, activation="relu")(x)
        x = Dropout(0.5)(x)

        output = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs=image_input, outputs=output)
        return model

    def _build_pretrained_model(
        self,
        image_shape: Tuple[int, int, int],
        param_shape: Tuple[int, ...],
        dropout1: float,
        dense1: int,
        dropout2: float,
        dense2: int,
        dropout3: float,
        backbone: Optional[str] = "mobilenet",
        unfreeze_from: Optional[int] = None,
    ) -> "tf.keras.Model":
        """
        Build a model using pre-trained MobileNetV2 to supplement small dataset.

        Args:
            image_shape: Dimensions of input images (height, width, channels)
            param_shape: Dimensions of numerical parameters
            unfreeze_from: Layer index from which to unfreeze weights (None = all frozen)
            dropout1: Perentage of inputs to zero out
            dense1: Number of neurons in first fully connected layer
            dropout2: Percentage of inputs to zero out
            dense2: Number of neurons in second fully connected layer
            dropout3: Percentage of final inputs to zero out

        Returns:
            tf.keras.Model: Compiled model ready for training
        """
        pre_trained_model = self._get_backbone_model(
            backbone=backbone, image_shape=image_shape
        )
        pre_trained_model.trainable = unfreeze_from is not None

        if unfreeze_from is not None:
            for layer in pre_trained_model.layers[:unfreeze_from]:
                layer.trainable = False

        # Data augmentation pipeline
        data_augmentation = Sequential(
            [
                RandomRotation(factor=0.0),
                RandomBrightness(factor=0.0),
                RandomZoom(height_factor=0.0, width_factor=0.0),
                GaussianNoise(stddev=0.00),
                RandomContrast(0.0),
            ]
        )

        # CNN for images
        image_input = Input(shape=image_shape)
        x = data_augmentation(image_input)
        x = pre_trained_model(x)
        x = GlobalAveragePooling2D(name="global_avg")(x)
        x = Dropout(dropout1, name="dropout")(x)

        # Numerical features section
        params_input = Input(shape=param_shape)
        y = Dense(dense1, name="dense_param1", activation="relu")(params_input)
        y = Dropout(dropout2, name="dropout_2")(
            y
        )  # Added to remove reliance on features
        y = BatchNormalization()(y)

        combined = Concatenate()([x, y])
        z = Dense(dense2, name="dense_combined", activation="relu")(combined)
        z = BatchNormalization(name="batch_norm")(z)
        z = Dropout(dropout3, name="dropout_combined")(z)
        if self.classification_type == "binary":
            activation = "sigmoid"
        elif self.classification_type == "multiclass":
            activation = "softmax"
        z = Dense(
            self.num_classes,
            name="output_layer",
            activation=activation,
        )(z)

        model = Model(inputs=[image_input, params_input], outputs=z)
        model.summary()
        return model

    def _build_image_only_model(
        self,
        image_shape: Tuple[int, int, int],
        backbone: Optional[str] = "mobilenet",
        num_classes: int = 5,
        dropout1_rate: Optional[float] = 0.1,
        dense_units: Optional[int] = 32,
        dropout2_rate: Optional[float] = 0.2,
        l2_factor: Optional[float] = None,
        unfreeze_from: Optional[int] = None,
    ) -> "tf.keras.Model":
        """
        Build a model that uses only the image input (no parameter features).

        Args:
            image_shape: Dimensions of input images
            backbone: Type of base model to use
            dropout1_rate: level of dropout for first layer
            dense_units: Units for first hidden layer
            dropout2_rate: Level of dropout for second layer

        Returns:
            tf.keras.Model: Image-only classification model
        """
        pre_trained_model = self._get_backbone_model(
            backbone=backbone, image_shape=image_shape
        )
        pre_trained_model.trainable = unfreeze_from is not None

        if unfreeze_from is not None:
            for layer in pre_trained_model.layers[:unfreeze_from]:
                layer.trainable = False

        data_augmentation = Sequential(
            [
                RandomRotation(factor=0.0),
                RandomBrightness(factor=0.0),
                RandomZoom(height_factor=0.0, width_factor=0.0),
                GaussianNoise(stddev=0.00),
                RandomContrast(0.0),
            ]
        )

        image_input = Input(shape=image_shape)
        x = data_augmentation(image_input)
        x = pre_trained_model(x)
        x = GlobalAveragePooling2D(name="global_avg")(x)
        x = Dropout(dropout1_rate, name="dropout")(x)
        if l2_factor:
            x = Dense(
                dense_units,
                name="dense1",
                activation="relu",
                kernel_regularizer=l2(l2_factor),
            )(x)
        else:
            x = Dense(dense_units, name="dense1", activation="relu")(x)

        x = Dropout(dropout2_rate, name="dropout_2")(x)
        if self.classification_type == "binary":
            activation = "sigmoid"
        elif self.classification_type == "multiclass":
            activation = "softmax"
        output = Dense(
            num_classes, name="output_layer", activation=activation
        )(x)

        model = Model(inputs=image_input, outputs=output)
        model.summary()
        return model

    def compile_image_only_model(
        self,
        image_shape: Tuple[int, int, int],
        learning_rate: float = 0.001,
        metrics: Optional[List[str]] = None,
        backbone: Optional[str] = "mobilenet",
        num_classes: int = 5,
        dropout1_rate: Optional[float] = 0.1,
        dense_units: Optional[int] = 32,
        dropout2_rate: Optional[float] = 0.2,
        l2_factor: Optional[float] = None,
        unfreeze_from: Optional[int] = None,
    ) -> "tf.keras.Model":
        """
        Compile an image-only model.

        Args:
            image_shape: Dimensions of input images
            learning_rate: Learning rate for optimization
            metrics: List of metrics to monitor

        Returns:
            tf.keras.Model: Compiled image-only model
        """
        if metrics is None:
            metrics = [
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ]

        model = self._build_image_only_model(
            image_shape,
            backbone=backbone,
            dropout1_rate=dropout1_rate,
            dense_units=dense_units,
            dropout2_rate=dropout2_rate,
            num_classes=num_classes,
            l2_factor=l2_factor,
            unfreeze_from=None,
        )
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        if self.classification_type == "binary":
            loss = "binary_crossentropy"
        elif self.classification_type == "multiclass":
            loss = "categorical_crossentropy"
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def compile_custom_model(
        self,
        image_shape: Tuple[int, int, int],
        learning_rate: float = 0.001,
        metrics: Optional[List[str]] = None,
        num_classes: Optional[int] = 5,
    ) -> "tf.keras.Model":
        """
        Compile custom model after calling build_custom_model function.

        Args:
            image_shape: Dimensions of input images
            param_shape: Dimensions of numerical parameters
            learning_rate: Learning rate for optimization
            metrics: List of metrics to monitor
            num_class: number of output classes

        Returns:
            tf.keras.Model: Compiled model ready for training
        """
        if metrics is None:
            metrics = ["accuracy"]

        model = self._build_custom_model(image_shape, num_classes=num_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if self.classification_type == "binary":
            loss = "binary_crossentropy"
        elif self.classification_type == "multiclass":
            loss = "categorical_crossentropy"
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def compile_model(
        self,
        image_shape: Tuple[int, int, int],
        param_shape: Tuple[int, ...],
        dropout1: float,
        dense1: int,
        dropout2: float,
        dense2: int,
        dropout3: float,
        learning_rate: float = 0.001,
        metrics: Optional[List[str]] = None,
        unfreeze_from: Optional[int] = None,
        backbone: Optional[str] = "mobilenet",
    ) -> "tf.keras.Model":
        """
        Compile model after calling build_model function.

        Args:
            image_shape: Dimensions of input images
            param_shape: Dimensions of numerical parameters
            learning_rate: Learning rate for optimization
            metrics: List of metrics to monitor
            unfreeze_from: Layer index from which to unfreeze weights

        Returns:
            tf.keras.Model: Compiled model ready for training
        """
        if metrics is None:
            metrics = [
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ]

        model = self._build_pretrained_model(
            image_shape,
            param_shape,
            unfreeze_from=unfreeze_from,
            backbone=backbone,
            dense1=dense1,
            dense2=dense2,
            dropout1=dropout1,
            dropout2=dropout2,
            dropout3=dropout3,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if self.classification_type == "binary":
            loss = "binary_crossentropy"
        elif self.classification_type == "multiclass":
            loss = "categorical_crossentropy"
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def create_checkpoints(
        self,
        checkpoint_filepath: str = "./checkpoints/model.keras",
        monitor: str = "val_accuracy",
        mode: str = "max",
        save_best_only: bool = True,
    ) -> ModelCheckpoint:
        """
        Create model checkpoints to avoid losing data while training.

        Args:
            checkpoint_filepath: Path to save model checkpoints
            monitor: Metric to monitor during training
            mode: Method to determine stopping point of metric (max, min, auto)
            save_best_only: Whether to save only the best model

        Returns:
            tf.keras.callbacks.ModelCheckpoint: Checkpoint callback
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)

        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only,
            verbose=1,
        )
        return model_checkpoint_callback

    def reduce_on_plateau(
        self,
        patience: int = 3,
        mode: str = "auto",
        factor: float = 2.0,
        monitor: str = "val_accuracy",
    ) -> ReduceLROnPlateau:
        """
        Create reduced learning rate callback if monitored value stops improving.

        Args:
            patience: Number of epochs before reducing learning rate
            mode: Method to monitor (min, max, auto)
            factor: Factor by which to reduce learning rate
            monitor: Value to monitor before reducing learning rate

        Returns:
            tf.keras.callbacks.ReduceLROnPlateau: Reduce learning rate callback
        """
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            patience=patience,
            mode=mode,
            factor=factor,
            min_lr=1e-7,
        )
        return reduce_lr

    def create_early_stopping(
        self,
        patience: int = 3,
        mode: str = "max",
        monitor: str = "val_accuracy",
    ) -> EarlyStopping:
        """
        Create early stopping callback to monitor training success and prevent overfitting.

        Args:
            patience: Number of epochs to wait before stopping when monitor plateaus
            mode: Method to track monitor (max, min, auto)
            monitor: Metric to monitor during training

        Returns:
            tf.keras.callbacks.EarlyStopping: Early stopping callback
        """
        es_callback = EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            restore_best_weights=True,
            verbose=1,
        )
        return es_callback

    def create_tensorboard_callback(
        self, log_dir: str = "./logs", histogram_freq: int = 1
    ) -> TensorBoard:
        """
        Create TensorBoard callback for monitoring training.

        Args:
            log_dir: Directory for TensorBoard logs
            histogram_freq: Frequency for computing weight histograms

        Returns:
            tf.keras.callbacks.TensorBoard: TensorBoard callback
        """
        return TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq)

    def train_model(
        self,
        class_weights,
        model: tf.keras.Model,
        epochs: int = 5,
        initial_epoch: int = 0,
        callbacks: Optional[List] = None,
        history_file: Optional[str] = None,
        save_model_file: Optional[str] = None,
    ) -> tf.keras.callbacks.History:
        """
        Train model with possible callbacks to prevent overfitting.

        Args:
            model: Model to be trained
            checkpoints: Checkpoints to save model
            epochs: Number of training epochs
            initial_epoch: Starting epoch number
            early_stopping: Early stopping callback
            reduce_lr: Reduce learning rate callback
            tensorboard: TensorBoard callback
            history_file: File to save training history
            model_file: File to save trained model

        Returns:
            tf.keras.callbacks.History: Training history
        """
        """callbacks = []
        
        if early_stopping:
            callbacks.append(early_stopping)
        if checkpoints:
            callbacks.append(checkpoints)
        if tensorboard:
            callbacks.append(tensorboard)
        if reduce_lr:
            callbacks.append(reduce_lr) """

        if callbacks:
            history = model.fit(
                self.train_ds,
                epochs=epochs,
                initial_epoch=initial_epoch,
                validation_data=self.test_ds,
                callbacks=callbacks,
                class_weight=class_weights,
            )
        else:
            print("Training without callbacks")
            history = model.fit(
                self.train_ds,
                epochs=epochs,
                initial_epoch=initial_epoch,
                validation_data=self.test_ds,
                class_weight=class_weights,
            )

        # Save training history
        if history_file:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            df = pd.DataFrame(history.history)
            df.to_csv(history_file, index=False)
            print(f"Training history saved to: {history_file}")
        else:
            print("Training history not saved")

        # Save trained model
        if save_model_file:
            os.makedirs(os.path.dirname(save_model_file), exist_ok=True)
            model.save(save_model_file)
            print(f"Model saved to: {save_model_file}")
        else:
            print("Model not saved")

        return history

    @staticmethod
    def train_kfold(
        datasets: List[Tuple],
        image_shape: Tuple[int, int, int],
        param_shape: Tuple[int, ...],
        learning_rate: float,
        metrics: List[str] = None,
        epochs: int = 5,
        initial_epoch: int = 0,
        history_file: Optional[str] = None,
        save_model_file: Optional[str] = None,
        callbacks: Optional[List] = None,
    ) -> Tuple[List[tf.keras.Model], List[tf.keras.callbacks.History]]:
        """
        Train model using k-fold cross validation.

        Args:
            datasets: List of (train_ds, test_ds) tuples for each fold
            image_shape: Dimensions of input images
            param_shape: Dimensions of numerical parameters
            learning_rate: Learning rate for optimization
            metrics: List of metrics to monitor
            epochs: Number of training epochs
            initial_epoch: Starting epoch number
            history_file: Base filename for saving training history
            model_file: Base filename for saving models

        Returns:
            Tuple of (list of trained models, list of training histories)
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall"]

        kfold_histories = []
        k_models = []
        train_datasets = [i[0] for i in datasets]
        test_datasets = [i[1] for i in datasets]

        for fold, (train_ds, test_ds) in enumerate(
            zip(train_datasets, test_datasets)
        ):
            print(f"\n=== Training fold {fold + 1} ===")

            custom_model = CustomModel(train_ds, test_ds)
            model = custom_model.compile_model(
                image_shape=image_shape,
                param_shape=param_shape,
                learning_rate=learning_rate,
                metrics=metrics,
            )
            es = EarlyStopping(
                monitor="val_accuracy",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            )

            callbacks = []

            if callbacks:
                history = model.fit(
                    train_ds,
                    epochs=epochs,
                    initial_epoch=initial_epoch,
                    validation_data=test_ds,
                    callbacks=callbacks,
                )
            else:
                print("Training without callbacks")
                history = model.fit(
                    train_ds,
                    epochs=epochs,
                    initial_epoch=initial_epoch,
                    validation_data=test_ds,
                )

            kfold_histories.append(history)
            k_models.append(model)

            # Save fold-specific history and model
            if history_file:
                os.makedirs(os.path.dirname(history_file), exist_ok=True)
                df = pd.DataFrame(history.history)
                df.to_csv(f"{history_file}_fold{fold+1}.csv", index=False)
                print(f"Fold {fold+1} history saved")
            else:
                print("History not saved")

            if save_model_file:
                os.makedirs(os.path.dirname(save_model_file), exist_ok=True)
                model.save(f"{save_model_file}_fold{fold+1}.keras")
                print(f"Fold {fold+1} model saved")
            else:
                print("Model not saved")

        return k_models, kfold_histories

    @staticmethod
    def get_averages_from_kfold(
        kfold_histories: List[tf.keras.callbacks.History],
    ) -> None:
        """
        Calculate and display average metrics from k-fold cross validation.

        Args:
            kfold_histories: List of training histories from k-fold training
        """
        accuracy = []
        precision = []
        recall = []

        for history in kfold_histories:
            accuracy.append(max(history.history["val_accuracy"]))
            precision.append(max(history.history["val_precision"]))
            recall.append(max(history.history["val_recall"]))

        avg_accuracy = np.mean(accuracy)
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)

        print(
            f"Average Accuracy: {avg_accuracy:.4f}, Std Dev: {np.std(accuracy):.4f}"
        )
        print(
            f"Average Precision: {avg_precision:.4f}, Std Dev: {np.std(precision):.4f}"
        )
        print(
            f"Average Recall: {avg_recall:.4f}, Std Dev: {np.std(recall):.4f}"
        )

    def plot_metric(
        self,
        title: str,
        metric_1: List[float],
        metric_2: List[float],
        metric_1_label: str,
        metric_2_label: str,
        x_label: str,
        y_label: str,
        model_path: str,
    ) -> None:
        """
        Plot training metrics for visualization.

        Args:
            title: Title for the plot
            metric_1: First metric values to plot
            metric_2: Second metric values to plot
            metric_1_label: Label for first metric
            metric_2_label: Label for second metric
            x_label: Label for x-axis
            y_label: Label for y-axis
        """
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.plot(metric_1, label=metric_1_label)
        plt.plot(metric_2, label=metric_2_label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        model_dir = os.path.dirname(model_path)
        basename = os.path.basename(model_path)
        stem, _ = os.path.splitext(basename)
        save_plot = os.path.join(model_dir, f"{stem}_{title}.png")
        plt.savefig(save_plot)
        plt.show()
