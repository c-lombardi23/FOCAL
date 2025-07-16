"""Test file for TestCNN command."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from cleave_app.commands.test_commands import TestCNN


@pytest.fixture
def test_cnn_config(tmp_path):
    config = MagicMock()
    config.cnn_mode = "bad_good"
    config.csv_path = str(tmp_path / "data.csv")
    config.img_folder = str(tmp_path / "images/")
    config.classification_path = str(tmp_path / "classification.csv")
    config.classification_type = "binary"
    config.model_path = str(tmp_path / "model.keras")
    config.backbone = "efficientnet"
    config.angle_threshold = 0.5
    config.diameter_threshold = 125.0
    config.feature_scaler_path = str(tmp_path / "scaler.pkl")
    config.num_classes = 1
    config.image_shape = [224, 224, 3]
    config.feature_shape = [5]

    return config


def test_test_cnn(mocker, test_cnn_config):

    config = test_cnn_config
    mock_test_predictions_class = mocker.patch(
        "cleave_app.commands.test_commands.TestPredictions"
    )
    mock_test_predictions_instance = mock_test_predictions_class.return_value
    mock_test_predictions_instance.gather_predictions.return_value = (
        np.array([1, 0, 1]),
        np.array([1, 1, 1]),
        np.array([0.34, 0.56, 0.68]),
    )

    mock_log_run = mocker.patch(
        "cleave_app.commands.test_commands.log_classifier_test_results"
    )
    mock_test_predictions_instance.display_confusion_matrix.return_value = (
        "confusion.png"
    )
    mock_test_predictions_instance.display_classification_report.return_value = (
        None
    )
    mock_test_predictions_instance.plot_roc.return_value = "roc.png"

    command = TestCNN()
    command._execute_command(config)

    mock_test_predictions_class.assert_called_once_with(
        model_path=config.model_path,
        csv_path=config.csv_path,
        scalar_path=config.feature_scaler_path,
        img_folder=config.img_folder,
        image_only=False,
        backbone=config.backbone,
        angle_threshold=config.angle_threshold,
        diameter_threshold=config.diameter_threshold,
        threshold=config.classification_threshold,
    )
    mock_test_predictions_instance.gather_predictions.assert_called_once()
    mock_test_predictions_instance.display_confusion_matrix.assert_called_once()
    mock_test_predictions_instance.display_classification_report.assert_called_once()
    mock_test_predictions_instance.plot_roc.assert_called_once()
    mock_log_run.assert_called_once()
