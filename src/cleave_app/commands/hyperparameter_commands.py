from cleave_app.data_processing import (
    DataCollector,
    MLPDataCollector
)
from cleave_app.hyperparameter_tuning import (
    HyperParameterTuning,
    MLPHyperparameterTuning,
    ImageHyperparameterTuning
)
import traceback
import pandas as pd

def _run_search_helper(
    config, tuner, train_ds, test_ds, best_params_path=None
) -> None:
    """Helper function for running hyperparameter search.

    Args:
        config: Configuration object
        tuner: Keras tuner instance
        train_ds: Training dataset
        test_ds: Test dataset
    """
    try:
        tuner.run_search(train_ds, test_ds)
        best_hp = tuner.get_best_hyperparameters().values
        print("Best hyperparameters:", best_hp)
        if best_params_path is not None:
            pd.DataFrame([best_hp]).to_csv(f"{best_params_path}")
        else:
            print("Best hyperparameters not saved")

        pathname = config.save_model_file
        if pathname is None:
            print("Model not saved - no path specified")
        else:
            if not pathname.endswith(".keras"):
                pathname = pathname + ".keras"
            tuner.save_best_model(pathname)
            print(f"Model saved to: {pathname}")

    except Exception as e:
        print(f"Error during hyperparameter search: {e}")
        traceback.print_exc()
        raise

class CNNHyperparameterSearch:

    def execute(self, config) -> None:
        """Perform hyperparameter search for CNN model.

        Args:
            config: Configuration object containing training parameters
        """
        try:
            data = DataCollector(
                config.csv_path,
                config.img_folder,
                backbone=config.backbone,
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
                feature_scaler_path=config.feature_scaler_path,
            )

            max_epochs = config.max_epochs or 20

            tuner = HyperParameterTuning(
                config.image_shape,
                config.feature_shape,
                max_epochs=max_epochs,
                project_name=config.project_name,
                directory=config.tuner_directory,
                backbone=config.backbone,
                class_weights=class_weights,
            )

            _run_search_helper(config, tuner, train_ds, test_ds)

        except Exception as e:
            print(f"Error during CNN hyperparameter tuning: {e}")
            traceback.print_exc()
            raise

class MLPHyperparameterSearch:

    def execute(self, config) -> None:
        """Perform hyperparameter search for MLP model.

        Args:
            config: Configuration object containing training parameters
        """
        try:
            data = MLPDataCollector(config.csv_path, config.img_folder)
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

            max_epochs = config.max_epochs or 20
            tuner = MLPHyperparameterTuning(
                config.model_path,
                max_epochs=max_epochs,
                project_name=config.project_name,
                directory=config.tuner_directory,
            )

            _run_search_helper(config, tuner, train_ds, test_ds)

        except Exception as e:
            print(f"Error during MLP hyperparameter tuning: {e}")
            traceback.print_exc()
            raise

class ImageHyperparameterSearch:
 
    def execute(self, config) -> None:
        """Perform hyperparameter search for image-only model.

        Args:
            config: Configuration object containing training parameters
        """

        try:
            data = DataCollector(
                config.csv_path,
                config.img_folder,
                backbone=config.backbone,
                set_mask=config.set_mask,
                classification_type=config.classification_type,
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

            max_epochs = config.max_epochs or 20
            tuner = ImageHyperparameterTuning(
                config.image_shape,
                max_epochs=max_epochs,
                project_name=config.project_name,
                directory=config.tuner_directory,
                backbone=config.backbone,
                num_classes=config.num_classes,
                class_weights=class_weights,
            )

            _run_search_helper(
                config,
                tuner,
                train_ds,
                test_ds,
                best_params_path=config.best_tuner_params,
            )

        except Exception as e:
            print(f"Error during image hyperparameter tuning: {e}")
            traceback.print_exc()
            raise
