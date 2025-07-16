"""Module for defining training logic to call from main entry point"""

import traceback

import tensorflow as tf

from cleave_app.data_processing import (BadCleaveTensionClassifier,
                                        DataCollector, MLPDataCollector)
from cleave_app.mlflow_utils import (log_cnn_training_run,
                                     log_image_training_run,
                                     log_mlp_training_run,
                                     log_xgb_training_run)
from cleave_app.mlp_model import BuildMLPModel
from cleave_app.model_pipeline import CustomModel
from cleave_app.xgboost_pipeline import XGBoostModel

from .base_command import BaseCommand
from .utils import _setup_callbacks


class TrainCNN(BaseCommand):
    """Train a CNN model for fiber cleave classification."""

    def _execute_command(self, config) -> None:
        if tf is None:
            raise ImportError("TensorFlow is required for CNN training")

        if config.cnn_mode == "bad_good":
            data = DataCollector(
                config.csv_path,
                config.img_folder,
                classification_type=config.classification_type,
                backbone=config.backbone,
                angle_threshold=config.angle_threshold,
                diameter_threshold=config.diameter_threshold,
            )
        elif config.cnn_mode == "tension":
            data = BadCleaveTensionClassifier(
                csv_path=config.csv_path,
                img_folder=config.img_folder,
                backbone=config.backbone,
                tension_threshold=config.tension_threshold,
            )
        else:
            raise ValueError(f"Unsupported cnn mode: {config.cnn_mode}")
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

        trainable_model = CustomModel(
            train_ds,
            test_ds,
            classification_type=config.classification_type,
            num_classes=config.num_classes,
        )

        if config.continue_train == "y":
            compiled_model = tf.keras.models.load_model(config.model_path)
        else:
            compiled_model = trainable_model.compile_model(
                image_shape=config.image_shape,
                param_shape=config.feature_shape,
                learning_rate=config.learning_rate or 0.001,
                unfreeze_from=config.unfreeze_from,
                backbone=config.backbone,
                dense1=config.dense1,
                dense2=config.dense2,
                dropout1=config.dropout1,
                dropout2=config.dropout2,
                dropout3=config.dropout3,
                brightness=config.brightness,
                contrast=config.contrast,
                height=config.height,
                width=config.width,
                rotation=config.rotation,
            )

            # Setup callbacks
        callbacks = _setup_callbacks(config, trainable_model)
        max_epochs = config.max_epochs or 20

        history = trainable_model.train_model(
            class_weights=class_weights,
            model=compiled_model,
            epochs=max_epochs,
            callbacks=callbacks,
            history_file=config.save_history_file,
            save_model_file=config.save_model_file,
        )

        log_cnn_training_run(
            config,
            compiled_model,
            history,
            dataset_path=config.csv_path,
            artifacts={
                "model": config.save_model_file,
                "history": config.save_history_file,
            },
        )

        # Plot training metrics
        trainable_model.plot_metric(
            "Loss vs. Val Loss",
            history.history["loss"],
            history.history["val_loss"],
            "loss",
            "val_loss",
            "epochs",
            "loss",
            model_path=config.save_model_file,
        )
        trainable_model.plot_metric(
            "Accuracy vs. Val Accuracy",
            history.history["accuracy"],
            history.history["val_accuracy"],
            "accuracy",
            "val_accuracy",
            "epochs",
            "accuracy",
            model_path=config.save_model_file,
        )


class TrainMLP(BaseCommand):
    """Train an MLP model for tension prediction."""

    def _execute_command(self, config) -> None:
        if tf is None:
            raise ImportError("TensorFlow is required for MLP training")

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
            feature_scaler_path=config.feature_scaler_path,
            tension_scaler_path=config.label_scaler_path,
        )

        trainable_model = BuildMLPModel(
            config.model_path,
            train_ds,
            test_ds,
            num_classes=config.num_classes,
        )

        # Setup callbacks
        callbacks = _setup_callbacks(config, trainable_model)
        max_epochs = config.max_epochs or 20

        compiled_model = trainable_model.compile_model(
            param_shape=config.feature_shape,
            learning_rate=config.learning_rate,
            dense1=config.dense1,
            dense2=config.dense2,
            dropout1=config.dropout1,
            dropout2=config.dropout2,
            dropout3=config.dropout3,
        )
        history = trainable_model.train_model(
            class_weights=None,
            model=compiled_model,
            epochs=max_epochs,
            callbacks=callbacks,
            history_file=config.save_history_file,
            save_model_file=config.save_model_file,
        )

        log_mlp_training_run(
            config,
            compiled_model,
            history,
            dataset_path=config.csv_path,
            artifacts={
                "model": config.save_model_file,
                "history": config.save_history_file,
            },
        )

        # Plot training metrics
        trainable_model.plot_metric(
            "Loss vs. Val Loss",
            history.history["loss"],
            history.history["val_loss"],
            "loss",
            "val_loss",
            "epochs",
            "loss",
            model_path=config.save_model_file,
        )
        trainable_model.plot_metric(
            "MAE vs. Val MAE",
            history.history["mae"],
            history.history["val_mae"],
            "mae",
            "val_mae",
            "epochs",
            "mae",
            model_path=config.save_model_file,
        )


class TrainXGBoost(BaseCommand):
    """Train an XGBoost model for predicting delta in tension"""

    def _execute_command(self, config):

        data = MLPDataCollector(
            csv_path=config.csv_path,
            img_folder=config.img_folder,
            backbone=None,
            angle_threshold=config.angle_threshold,
            diameter_threshold=config.diameter_threshold,
        )

        images, features, labels = data.extract_data()
        train_ds, test_ds = data.create_datasets(
            images,
            features,
            labels,
            test_size=config.test_size,
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
            feature_scaler_path=None,
            tension_scaler_path=config.label_scaler_path,
        )

        train_ds = data.image_only_dataset(train_ds)
        test_ds = data.image_only_dataset(test_ds)

        xgb_model = XGBoostModel(
            csv_path=config.csv_path,
            cnn_model_path=config.model_path,
            train_ds=train_ds,
            test_ds=test_ds,
        )

        evals_result = xgb_model.train(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            random_state=config.random_state,
            gamma=config.gamma,
            subsample=config.subsample,
            reg_lambda=config.reg_lambda,
        )
        xgb_model.save(config.xgb_path)

        X_train, y_train = xgb_model._extract_features_and_labels(train_ds)

        log_xgb_training_run(
            config=config,
            model=xgb_model.get_model(),
            evals_result=evals_result,
            X_train=X_train,
            y_train=y_train,
            dataset_path=config.csv_path,
            artifacts={"model": config.xgb_path},
        )

        xgb_model.plot_metrics(
            title="RMSE vs. Val RMSE",
            metric1=evals_result["validation_0"]["rmse"],
            metric2=evals_result["validation_1"]["rmse"],
            metric1_label="RSME",
            metric2_label="Val RMSE",
            x_label="Training Round",
            y_label="RMSE",
        )


class TrainImageOnly(BaseCommand):
    """Train the CNN model with only images."""

    def _execute_command(self, config) -> None:

        if tf is None:
            raise ImportError("TensorFlow is required for image-only training")

        try:
            data = DataCollector(
                config.csv_path,
                config.img_folder,
                classification_type=config.classification_type,
                backbone=config.backbone,
                set_mask=config.set_mask,
                encoder_path=config.encoder_path,
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

            trainable_model = CustomModel(
                train_ds,
                test_ds,
                classification_type=config.classification_type,
                num_classes=config.num_classes,
            )
            if config.continue_train == "y":
                compiled_model = tf.keras.models.load_model(config.model_path)
            else:
                compiled_model = trainable_model.compile_image_only_model(
                    config.image_shape,
                    config.learning_rate or 0.001,
                    backbone=config.backbone,
                    dropout1=config.dropout1,
                    dense1=config.dense1,
                    dropout2=config.dropout2,
                    l2_factor=config.l2_factor,
                    num_classes=config.num_classes,
                    unfreeze_from=config.unfreeze_from,
                )
            # Setup callbacks
            callbacks = _setup_callbacks(config, trainable_model)
            max_epochs = config.max_epochs or 20

            history = trainable_model.train_model(
                class_weights,
                compiled_model,
                epochs=max_epochs,
                callbacks=callbacks,
                history_file=config.save_history_file,
                save_model_file=config.save_model_file,
            )

            log_image_training_run(
                config,
                compiled_model,
                history,
                dataset_path=config.csv_path,
                artifacts={
                    "model": config.save_model_file,
                    "history": config.save_history_file,
                },
            )

            # Plot training metrics
            trainable_model.plot_metric(
                "Loss vs. Val Loss",
                history.history["loss"],
                history.history["val_loss"],
                "loss",
                "val_loss",
                "epochs",
                "loss",
                model_path=config.save_model_file,
            )
            trainable_model.plot_metric(
                "Accuracy vs. Val Accuracy",
                history.history["accuracy"],
                history.history["val_accuracy"],
                "accuracy",
                "val_accuracy",
                "epochs",
                "accuracy",
                model_path=config.save_model_file,
            )

        except Exception as e:
            print(f"Error during image-only training: {e}")
            traceback.print_exc()
            raise


class KFoldCNN(BaseCommand):
    """Train CNN model using k-fold cross validation."""

    def _execute_command(self, config) -> None:
        data = DataCollector(
            config.csv_path,
            config.img_folder,
            backbone=config.backbone,
            angle_threshold=config.angle_threshold,
            diameter_threshold=config.diameter_threshold,
        )
        images, features, labels = data.extract_data()
        datasets = data.create_kfold_datasets(
            images,
            features,
            labels,
            config.buffer_size,
            config.batch_size,
        )

        _, kfold_histories = CustomModel.train_kfold(
            datasets,
            config.image_shape,
            config.feature_shape,
            config.learning_rate or 0.001,
            num_classes=config.num_classes,
            epochs=config.max_epochs,
            history_file=config.save_history_file,
            save_model_file=config.save_model_file,
            dense1=config.dense1,
            dense2=config.dense2,
            dropout1=config.dropout1,
            dropout2=config.dropout2,
            dropout3=config.dropout3,
            brightness=config.brightness,
            contrast=config.contrast,
            height=config.height,
            width=config.width,
            rotation=config.rotation,
        )

        CustomModel.get_averages_from_kfold(kfold_histories)


class KFoldMLP(BaseCommand):
    """Train MLP model using k-fold cross validation."""

    def _execute_command(self, config) -> None:
        data = MLPDataCollector(
            config.csv_path,
            config.img_folder,
            backbone=None,
            angle_threshold=config.angle_threshold,
            diameter_threshold=config.diameter_threshold,
        )
        images, features, labels = data.extract_data()
        datasets, scaler = data.create_kfold_datasets(
            images,
            features,
            labels,
            config.buffer_size,
            config.batch_size,
        )

        kfold_histories = BuildMLPModel.train_kfold_mlp(
            datasets,
            config.model_path,
            config.feature_shape,
            config.learning_rate or 0.001,
            dense1=config.dense1,
            dense2=config.dense2,
            dropout1=config.dropout1,
            dropout2=config.dropout2,
            dropout3=config.dropout3,
            num_classes=config.num_classes,
            history_file=config.save_history_file,
            save_model_file=config.save_model_file,
        )

        BuildMLPModel.get_averages_from_kfold(kfold_histories, scaler)


class TrainCustomModel(BaseCommand):
    """Train a custom CNN model without pre-trained base."""

    def _execute_command(self, config) -> None:
        data = DataCollector(
            config.csv_path,
            config.img_folder,
            angle_threshold=config.angle_threshold,
            diameter_threshold=config.diameter_threshold,
        )
        train_ds, test_ds = data.create_custom_dataset(
            config.image_shape,
            config.test_size,
            config.buffer_size,
            config.batch_size,
        )
        trainable_model = CustomModel(train_ds, test_ds)

        compiled_model = trainable_model.compile_custom_model(
            config.image_shape, config.learning_rate
        )

        callbacks = _setup_callbacks(config, trainable_model)
        max_epochs = config.max_epochs or 20

        history = trainable_model.train_model(
            model=compiled_model,
            epochs=max_epochs,
            class_weights=None,
            callbacks=callbacks,
            history_file=config.save_history_file,
            save_model_file=config.save_model_file,
        )

        # Plot training metrics
        trainable_model.plot_metric(
            "Loss vs. Val Loss",
            history.history["loss"],
            history.history["val_loss"],
            "loss",
            "val_loss",
            "epochs",
            "loss",
            model_path=config.save_model_file,
        )
        trainable_model.plot_metric(
            "Accuracy vs. Val Accuracy",
            history.history["accuracy"],
            history.history["val_accuracy"],
            "accuracy",
            "val_accuracy",
            "epochs",
            "accuracy",
            model_path=config.save_model_file,
        )
