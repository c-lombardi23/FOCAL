"""This module define the logic for testing each model."""

import traceback

from cleave_app.mlflow_utils import log_cnn_test_results
from cleave_app.prediction_testing import (TensionPredictor, TestPredictions,
                                           TestTensionPredictions)
from cleave_app.xgboost_pipeline import XGBoostPredictor

from .base_command import BaseCommand


class TestCNN(BaseCommand):
    """Test CNN model performance."""

    def _execute_command(self, config) -> None:
        if config.cnn_mode == "good_bad":
            tester = TestPredictions(
                config.model_path,
                config.csv_path,
                config.feature_scaler_path,
                config.img_folder,
                image_only=False,
                backbone=config.backbone,
                angle_threshold=config.angle_threshold,
                diameter_threshold=config.diameter_threshold,
            )
        elif config.cnn_mode == "tension":
            tester = TestTensionPredictions(
                cnn_model_path=config.model_path,
                tension_model_path=config.tension_model_path,
                csv_path=config.csv_path,
                scaler_path=config.feature_scaler_path,
                img_folder=config.img_folder,
                tension_threshold=config.tension_threshold,
                image_only=False,
            )

        true_labels, pred_labels, predictions = tester.gather_predictions()

        if true_labels is not None:
            confusion_Path = tester.display_confusion_matrix(
                true_labels, pred_labels, model_path=config.model_path
            )

            tester.display_classification_report(
                true_labels, pred_labels, config.classification_path
            )
            roc_path = tester.plot_roc(
                "ROC Curve",
                true_labels=true_labels,
                pred_probabilites=predictions,
            )
            log_cnn_test_results(
                tester=tester,
                run_name="cnn_test_results",
                confusion_matrix_path=confusion_Path,
                model_path=config.model_path,
                classification_path=config.classification_path,
                roc_path=roc_path,
                true_labels=true_labels,
                pred_labels=pred_labels,
                predictions=predictions,
            )

        else:
            print("No predictions generated - check data paths")


class TestMLP(BaseCommand):
    """Test MLP model performance."""

    def _execute_command(self, config) -> None:

        predictor = TensionPredictor(
            model_path=config.model_path,
            image_folder=config.img_folder,
            tension_scaler_path=config.label_scaler_path,
            feature_scaler_path=config.feature_scaler_path,
            csv_path=config.csv_path,
        )
        """
        if config.test_features is not None:
            prediction = predictor.PredictTension(config.test_features)
            print(f"Predicted tension: {prediction}")
        else:
            print("No test features provided")
        """
        predictor.find_best_tension_for_image([100, 200])


class TestImageOnly(BaseCommand):
    """Test CNN model performance on only images."""

    def execute(self, config) -> None:
        tester = TestPredictions(
            model_path=config.model_path,
            csv_path=config.csv_path,
            img_folder=config.img_folder,
            scalar_path=None,
            image_only=True,
            backbone=config.backbone,
            encoder_path=config.encoder_path,
            classification_type=config.classification_type,
        )
        true_labels, pred_labels = tester.gather_predictions()

        if true_labels is not None:
            tester.display_confusion_matrix(
                true_labels, pred_labels, model_path=config.model_path
            )
            tester.display_classification_report(
                true_labels, pred_labels, config.classification_path
            )
        else:
            print("No predictions generated - check data paths")


class TestXGBoost(BaseCommand):
    """Test XGBoost performance on prediciting change in tension."""

    def _execute_command(self, config):
        xgb_predicter = XGBoostPredictor(
            xgb_path=config.xgb_path,
            csv_path=config.csv_path,
            angle_threshold=config.angle_threshold,
            diameter_threshold=config.diameter_threshold,
            scaler_path=config.label_scaler_path,
            cnn_model_path=config.model_path,
        )
        xgb_predicter.load()
        xgb_predicter.predict()
