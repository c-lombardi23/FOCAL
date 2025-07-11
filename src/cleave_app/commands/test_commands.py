from cleave_app.prediction_testing import (
    TestPredictions,
    TestTensionPredictions,
    TensionPredictor
)
from cleave_app.mlflow_utils import (
    log_cnn_test_results
)
from cleave_app.xgboost_pipeline import XGBoostPredictor
import traceback

class TestCNN:

    
    def execute(self, config) -> None:
        """Test CNN model performance.

        Args:
            config: Configuration object containing test parameters
        """
        try:
            if config.cnn_mode == "good_bad":
                tester = TestPredictions(
                    config.model_path,
                    config.csv_path,
                    config.feature_scaler_path,
                    config.img_folder,
                    image_only=False,
                    backbone=config.backbone,
                    angle_threshold=config.angle_threshold,
                    diameter_threshold=config.diameter_threshold
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
                log_cnn_test_results(tester=tester,
                                    run_name="cnn_test_results",
                                    confusion_matrix_path=confusion_Path,
                                    model_path=config.model_path,
                                    classification_path=config.classification_path,
                                    roc_path=roc_path,
                                    true_labels=true_labels,
                                    pred_labels=pred_labels,
                                    predictions=predictions)

            else:
                print("No predictions generated - check data paths")

        except Exception as e:
            print(f"Error during CNN testing: {e}")
            traceback.print_exc()
            raise

class TestMLP:

    def execute(self, config) -> None:
        """Test MLP model performance.

        Args:
            config: Configuration object containing test parameters
        """
        try:
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
        except Exception as e:
            print(f"Error during MLP testing: {e}")
            traceback.print_exc()
            raise

class TestImageOnly:

    def execute(self, config) -> None:
        """Test CNN model performance on only images.

        Args:
            config: Configuration object containing test parameters
        """

        try:
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

        except Exception as e:
            print(f"Error during CNN testing: {e}")
            traceback.print_exc()
            raise

class TestXGBoost:

    def execute(self, config):
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
