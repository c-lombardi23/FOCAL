"""This module define the logic for testing each model."""

from cleave_app.mlflow_utils import (log_classifier_test_results,
                                     log_regressor_test_results)
from cleave_app.prediction_testing import (TensionPredictor, TestPredictions,
                                           TestTensionPredictions)
from cleave_app.xgboost_pipeline import XGBoostPredictor

from .base_command import BaseCommand


class TestCNN(BaseCommand):
    """Test CNN model performance."""

    def _execute_command(self, config) -> None:
        if config.cnn_mode == "bad_good":
            tester = TestPredictions(
                model_path=config.model_path,
                csv_path=config.csv_path,
                scalar_path=config.feature_scaler_path,
                img_folder=config.img_folder,
                image_only=False,
                backbone=config.backbone,
                angle_threshold=config.angle_threshold,
                diameter_threshold=config.diameter_threshold,
                threshold=config.classification_threshold
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
            log_classifier_test_results(
                tester=tester,
                run_name="cnn_test_results",
                confusion_matrix_path=confusion_Path,
                model_path=config.model_path,
                dataset_path=config.csv_path,
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
            csv_path=config.csv_path,
            angle_threshold=config.angle_threshold,
            diameter_threshold=config.diameter_threshold,
        )

        tensions, true_delta, predicted_deltas, predictions = (
            predictor.predict()
        )

        log_regressor_test_results(
            model_path=config.model_path,
            run_name="mlp_results",
            experiment_name="mlp_results",
            dataset_path=config.csv_path,
            tensions=tensions,
            predicted_delta=predicted_deltas,
            predictions=predictions,
            true_delta=true_delta,
        )


class TestImageOnly(BaseCommand):
    """Test CNN model performance on only images."""

    def _execute_command(self, config) -> None:
        tester = TestPredictions(
            model_path=config.model_path,
            csv_path=config.csv_path,
            img_folder=config.img_folder,
            scalar_path=None,
            image_only=True,
            backbone=config.backbone,
            encoder_path=config.encoder_path,
            classification_type=config.classification_type,
            angle_threshold=config.angle_threshold,
            diameter_threshold=config.diameter_threshold,
        )
        true_labels, pred_labels, predictions = tester.gather_predictions()

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
            log_classifier_test_results(
                tester=tester,
                run_name="image_only_test_results",
                confusion_matrix_path=confusion_Path,
                model_path=config.model_path,
                dataset_path=config.csv_path,
                classification_path=config.classification_path,
                roc_path=roc_path,
                true_labels=true_labels,
                pred_labels=pred_labels,
                predictions=predictions,
                image_only=True,
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
        tensions, predicted_deltas, predictions, true_delta = (
            xgb_predicter.predict()
        )

        log_regressor_test_results(
            model_path=config.xgb_path,
            run_name="xgb_results",
            experiment_name="xgb_results",
            dataset_path=config.csv_path,
            tensions=tensions,
            predicted_delta=predicted_deltas,
            predictions=predictions,
            true_delta=true_delta,
        )
