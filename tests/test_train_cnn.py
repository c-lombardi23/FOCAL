"""Test file for the TrainCNN command"""

from unittest.mock import MagicMock, ANY

import numpy as np
import pytest
import tensorflow as tf

from cleave_app.commands.train_commands import TrainCNN, TrainImageOnly


@pytest.fixture
def train_cnn_config(tmp_path):

    config = MagicMock()
    config.cnn_mode = "bad_good"
    config.csv_path = str(tmp_path / "data.csv")
    config.img_folder = str(tmp_path / "images/")
    config.classification_type = "binary"
    config.backbone = "efficientnet"
    config.angle_threshold = 0.5
    config.diameter_threshold = 125.0
    config.test_size = 0.2
    config.buffer_size = 100
    config.batch_size = 8
    config.feature_scaler_path = str(tmp_path / "scaler.pkl")
    config.num_classes = 1
    config.continue_train = "n"
    config.image_shape = [224, 224, 3]
    config.feature_shape = [5]
    config.learning_rate = 0.001
    config.unfreeze_from = None
    config.max_epochs = 2
    config.save_history_file = str(tmp_path / "history.csv")
    config.save_model_file = str(tmp_path / "model.keras")
    config.encoder_path = None
    config.test_p =0.9
    config.train_p = 1.0

    config.dense1 = 32
    config.dense2 = 16
    config.dropout1 = 0.2
    config.dropout2 = 0.2
    config.dropout3 = 0.2
    config.l2_factor = 0.1
    config.brightness = 0.1
    config.contrast = 0.1
    config.height = 0.1
    config.width = 0.1
    config.rotation = 0.1
    config.monitor = "val_loss"
    config.method = "min"
    config.unfreeze_from = None

    config.set_mask = "y"

    return config


def run_training_command(mocker, config, command_class, image_only=True):

    mock_images = np.zeros((10, 224, 224, 3))
    mock_features = np.zeros((10, 5))
    mock_labels = np.zeros((10, 1))

    mock_train_ds = tf.data.Dataset.from_tensor_slices(
        ((mock_images, mock_features), mock_labels)
    ).batch(config.batch_size)
    mock_test_ds = tf.data.Dataset.from_tensor_slices(
        ((mock_images, mock_features), mock_labels)
    ).batch(config.batch_size)
    mock_class_weights = {0: 1.0}

    data_collector = mocker.patch(
        "cleave_app.commands.train_commands.DataCollector"
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
        mock_class_weights,
    )

    if image_only:
        mock_train_ds = tf.data.Dataset.from_tensor_slices(
            (mock_images, mock_labels)
        ).batch(config.batch_size)
        mock_test_ds = tf.data.Dataset.from_tensor_slices(
            (mock_images, mock_labels)
        ).batch(config.batch_size)
        data_collector_instance.image_only_dataset.side_effect = [
            mock_train_ds,
            mock_test_ds,
        ]

    mock_history = MagicMock()
    mock_history.history = {
        "loss": [0.5],
        "val_loss": [0.6],
        "accuracy": [0.8],
        "val_accuracy": [0.75],
    }

    mock_custom_model = mocker.patch(
        "cleave_app.commands.train_commands.CustomModel"
    )
    mock_custom_model_instance = mock_custom_model.return_value
    if image_only:
        mock_custom_model_instance.compile_image_only_model.return_value = (
            mock_custom_model_instance.compile_model
        )
    else:
        mock_custom_model_instance.compile_model.return_value = (
            mock_custom_model_instance.compile_model
        )
    mock_custom_model_instance.train_model.return_value = mock_history

    mocker.patch(
        "cleave_app.commands.train_commands._setup_callbacks", return_value=[]
    )

    if image_only:
        mock_log_run = mocker.patch(
            "cleave_app.commands.train_commands.log_image_training_run"
        )
    else:
        mock_log_run = mocker.patch(
            "cleave_app.commands.train_commands.log_cnn_training_run"
        )

    command = command_class()
    command._execute_command(config)

    data_collector.assert_called_once()

    '''if image_only:
        data_collector_instance.create_datasets.assert_called_once_with(
            ANY,
            ANY,
            ANY,
            config.test_size,
            config.buffer_size,
            config.batch_size,
            config.train_p,
            config.test_p,
        )
        assert data_collector_instance.image_only_dataset.call_count == 2
    else:
        data_collector_instance.create_datasets.assert_called_once_with(
            ANY,
            ANY,
            ANY,
            config.test_size,
            config.buffer_size,
            config.batch_size,
            config.train_p,
            config.test_p,
            feature_scaler_path=config.feature_scaler_path,
            
        )'''

    mock_custom_model.assert_called_once_with(
        mock_train_ds,
        mock_test_ds,
        classification_type=config.classification_type,
        num_classes=config.num_classes,
    )

    if image_only:
        mock_custom_model_instance.compile_image_only_model.assert_called_once_with(
            config.image_shape,
            config.learning_rate,
            backbone=config.backbone,
            dropout1=config.dropout1,
            dense1=config.dense1,
            dropout2=config.dropout2,
            l2_factor=config.l2_factor,
            num_classes=config.num_classes,
            unfreeze_from=config.unfreeze_from,
        )
    else:
        mock_custom_model_instance.compile_model.assert_called_once_with(
            image_shape=config.image_shape,
            param_shape=config.feature_shape,
            learning_rate=config.learning_rate,
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

    mock_custom_model_instance.train_model.assert_called_once()

    assert mock_custom_model_instance.plot_metric.call_count == 2
    mock_log_run.assert_called_once()


def test_train_cnn(mocker, train_cnn_config):
    command_class = TrainCNN
    run_training_command(
        mocker, train_cnn_config, command_class, image_only=False
    )


def test_train_image_only(mocker, train_cnn_config):
    command_class = TrainImageOnly
    run_training_command(
        mocker, train_cnn_config, command_class, image_only=True
    )
