"""Test file for TrainXGBoost command."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from cleave_app.commands.train_commands import TrainXGBoost


@pytest.fixture
def train_xgboost_config(tmp_path):

    cnn_input = tf.keras.Input(shape=(224, 224, 3), name="cnn_input")
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")(cnn_input)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg")(x)
    cnn_output = tf.keras.layers.Dense(8)(x)
    cnn_model = tf.keras.Model(inputs=cnn_input, outputs=cnn_output)

    model_path = str(tmp_path / "cnn_model.keras")
    cnn_model.save(model_path)

    config = MagicMock()
    config.csv_path = "fake/data.csv"
    config.img_folder = "fake/images/"
    config.classification_type = "binary"
    config.backbone = "efficientnet"
    config.angle_threshold = 0.5
    config.diameter_threshold = 125.0
    config.model_path = model_path
    config.test_size = 0.2
    config.buffer_size = 100
    config.batch_size = 8
    config.label_scaler_path = str(tmp_path / "label_scaler.pkl")
    config.image_shape = [224, 224, 3]
    config.feature_shape = [4]
    config.learning_rate = 0.001
    config.max_epochs = 2
    config.save_history_file = str(tmp_path / "history.csv")
    config.xgb_path = str(tmp_path / "xgb_model.keras")

    config.error_type = "reg:squarederror"
    config.n_estimators = 100
    config.max_depth = 4
    config.random_state = 42
    config.gamma = 0.1
    config.subsample = 0.8
    config.reg_lambda = 2.0

    return config


def test_train_xgb(mocker, train_xgboost_config):

    config = train_xgboost_config

    mock_images = np.zeros((10, 224, 224, 3))
    mock_features = np.zeros((10, 5))
    mock_labels = np.zeros((10, 1))
    mock_train_ds = tf.data.Dataset.from_tensor_slices(
        ((mock_images, mock_features), mock_labels)
    ).batch(8)
    mock_test_ds = tf.data.Dataset.from_tensor_slices(
        ((mock_images, mock_features), mock_labels)
    ).batch(8)
    mock_class_weights = {0: 1.0}

    data_collector = mocker.patch(
        "cleave_app.commands.train_commands.MLPDataCollector"
    )

    data_collector_instance = data_collector.return_value
    data_collector_instance.extract_data.return_value = (
        mock_images,
        mock_features,
        mock_labels,
    )
    data_collector_instance.create_datasets.return_value = (
        mock_train_ds,
        mock_test_ds,
    )
    data_collector_instance.image_only_dataset.side_efect = lambda ds: ds

    mock_xgb_class = mocker.patch(
        "cleave_app.commands.train_commands.XGBoostModel"
    )
    xgb_model_instance = mock_xgb_class.return_value
    xgb_model_instance.train.return_value = {
        "validation_0": {"rmse": [0.1, 0.05]},
        "validation_1": {"rmse": [0.12, 0.07]},
    }

    xgb_model_instance._extract_features_and_labels.return_value = (
        mock_features,
        mock_labels,
    )
    xgb_model_instance.get_model.return_value = "mock_model"

    mock_log_run = mocker.patch(
        "cleave_app.commands.train_commands.log_xgb_training_run"
    )

    command = TrainXGBoost()
    command._execute_command(config)

    data_collector.assert_called_once_with(
        csv_path=config.csv_path,
        img_folder=config.img_folder,
        backbone=None,
        angle_threshold=config.angle_threshold,
        diameter_threshold=config.diameter_threshold,
    )

    data_collector_instance.create_datasets.assert_called_once()
    assert data_collector_instance.image_only_dataset.call_count == 2

    mock_xgb_class.assert_called_once()
    xgb_model_instance.train.assert_called_once()
    xgb_model_instance.save.assert_called_once_with(config.xgb_path)

    mock_log_run.assert_called_once()
