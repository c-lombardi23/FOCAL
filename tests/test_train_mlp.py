"""Test file for TrainMLP command"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock

from cleave_app.commands.train_commands import TrainMLP

@pytest.fixture
def train_mlp_config(tmp_path):

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
    config.feature_scaler_path = str(tmp_path / "feature_scaler.pkl")
    config.label_scaler_path = str(tmp_path /"label_scaler.pkl")
    config.image_shape = [224, 224, 3]
    config.feature_shape = [4]
    config.learning_rate = 0.001
    config.max_epochs = 2 
    config.save_history_file = str(tmp_path / "history.csv")
    config.save_model_file = str(tmp_path / "model.keras")
    
    config.dense1=32
    config.dense2=16
    config.dropout1=0.2
    config.dropout2=0.2
    config.dropout3=0.2
    config.brightness=0.1
    config.contrast=0.1
    config.height=0.1
    config.width=0.1
    config.rotation=0.1
    config.monitor = "val_loss"
    config.method = "min"

    return config

def test_train_ml(mocker, train_mlp_config):

    config = train_mlp_config

    mock_images = np.zeros((10, 224, 224, 3))
    mock_features = np.zeros((10, 5))
    mock_labels = np.zeros((10, 1))
    mock_train_ds = tf.data.Dataset.from_tensor_slices(((mock_images, mock_features), mock_labels)).batch(8)
    mock_test_ds = tf.data.Dataset.from_tensor_slices(((mock_images, mock_features), mock_labels)).batch(8)
    mock_class_weights = {0: 1.0}

    data_collector = mocker.patch("cleave_app.commands.train_commands.MLPDataCollector")

    data_collector_instance = data_collector.return_value
    data_collector_instance.extract_data.return_value = (mock_images, mock_features, mock_labels)
    data_collector_instance.create_datasets.return_value = (mock_train_ds, mock_test_ds)

    mock_history = MagicMock()
    mock_history.history = {
        'loss': [0.5],
        'val_loss': [0.6],
        'mae': [0.08],
        'val_mae': [0.075]
    }

    mock_custom_model = mocker.patch("cleave_app.commands.train_commands.BuildMLPModel")
    mock_custom_model_instance = mock_custom_model.return_value
    mock_custom_model_instance.compile_model.return_value = mock_custom_model_instance.compile_model
    mock_custom_model_instance.train_model.return_value = mock_history

    mocker.patch("cleave_app.commands.train_commands._setup_callbacks", return_value=[])
    mock_log_run = mocker.patch("cleave_app.commands.train_commands.log_mlp_training_run")

    command = TrainMLP()
    command._execute_command(config)

    data_collector.assert_called_once_with(
        config.csv_path,
        config.img_folder, 
        angle_threshold = config.angle_threshold,
        diameter_threshold = config.diameter_threshold
    )

    data_collector_instance.create_datasets.assert_called_once()

    mock_custom_model.assert_called_once()
    mock_custom_model_instance.compile_model.assert_called_once()
    mock_custom_model_instance.train_model.assert_called_once()

    assert mock_custom_model_instance.plot_metric.call_count == 2
    mock_log_run.assert_called_once()