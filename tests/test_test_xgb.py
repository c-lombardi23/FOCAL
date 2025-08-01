"""Test file for TestXGBoost command."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from cleave_app.commands.test_commands import TestXGBoost


@pytest.fixture
def test_xgb_config(tmp_path):

    config = MagicMock()
    config.csv_path = str(tmp_path / "data.csv")
    config.img_folder = str(tmp_path / "images/")
    config.mode = "test_xgboost"
    config.image_shape = [224, 224, 3]
    config.set_mask = "y"
    config.feature_shape = [4]

    config.angle_threshold = 0.55
    config.diameter_threshold = 56
    config.label_scaler_path = str(tmp_path / "label_scaler.pkl")
    config.model_path = str(tmp_path / "model.keras")

    config.xgb_path = str(tmp_path / "xgb.pkl")

    return config


def test_test_xgb(mocker, test_xgb_config):
    config = test_xgb_config

    mock_xgb_predictor = mocker.patch(
        "cleave_app.commands.test_commands.XGBoostPredictor"
    )
    mock_tester = mock_xgb_predictor.return_value

    mock_tester.predict.return_value = (
        np.array([112.0, 120, 145]),
        np.array([23.45, -32.5, 10.0]),
        np.array([200, 134, 165]),
        np.array([10.6, -45.6, 120.34]),
        43.5
    )
    mock_log_run = mocker.patch(
        "cleave_app.commands.test_commands.log_regressor_test_results"
    )

    command = TestXGBoost()
    command._execute_command(config)

    mock_xgb_predictor.assert_called_once_with(
        xgb_path=config.xgb_path,
        csv_path=config.csv_path,
        angle_threshold=config.angle_threshold,
        diameter_threshold=config.diameter_threshold,
        scaler_path=config.label_scaler_path,
        cnn_model_path=config.model_path,
    )
    mock_tester.load.assert_called_once()
    mock_tester.predict.assert_called_once()
    mock_log_run.assert_called_once()
