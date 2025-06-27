"""
Main CLI module for the Fiber Cleave Processing application.

This module provides the command-line interface for training and testing
CNN and MLP models for fiber cleave quality classification and tension prediction.
"""

import warnings
import os
import argparse
from typing import Optional

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Import application modules
from .config_schema import load_config 
from .data_processing import DataCollector, MLPDataCollector
from .model_pipeline import CustomModel, BuildMLPModel
from .prediction_testing import TestPredictions, TensionPredictor
from .hyperparameter_tuning import (
    HyperParameterTuning, 
    ImageHyperparameterTuning, 
    MLPHyperparameterTuning
)
from .grad_cam import gradcam_driver, compute_saliency_map
import pandas as pd
from typing import List, Callable

try:
    import tensorflow as tf
except ImportError:
    print("Warning: TensorFlow not found. Please install tensorflow>=2.19.0")
    tf = None

def _setup_callbacks(config, trainable_model) -> List[Callable]:
    """
    Setup training callbacks based on configuration.
    
    Args:
        config: Configuration object containing callback parameters
        trainable_model: Model instance that has callback creation methods
        
    Returns:
        List[Callable]: List of configured callbacks
    """
    callbacks = []
    
    # Setup checkpoint callback
    if (config.checkpoints == "y" and 
        config.checkpoint_filepath and 
        config.monitor and 
        config.method):
        checkpoint = trainable_model.create_checkpoints(
            config.checkpoint_filepath, 
            config.monitor, 
            config.method
        )
        callbacks.append(checkpoint)
    
    # Setup early stopping callback
    if (config.early_stopping == "y" and 
        config.patience and 
        config.monitor and 
        config.method):
        es = trainable_model.create_early_stopping(
            config.patience, 
            config.method, 
            config.monitor
        )
        callbacks.append(es)
    
    # Setup learning rate reduction callback
    if config.reduce_lr is not None:
        if config.reduce_lr_patience is not None:
            reduce_lr = trainable_model.reduce_on_plateau(
                patience=config.reduce_lr_patience, 
                factor=config.reduce_lr
            )
        else:
            reduce_lr = trainable_model.reduce_on_plateau(
                factor=config.reduce_lr
            )
        callbacks.append(reduce_lr)
    
    return callbacks


def _train_cnn(config) -> None:
    """
    Train a CNN model for fiber cleave classification.

    Args:
        config: Configuration object containing training parameters
    """
    if tf is None:
        raise ImportError("TensorFlow is required for CNN training")
        
    try:
        data = DataCollector(config.csv_path, config.img_folder, classification_type=config.classification_type, backbone=config.backbone)
        images, features, labels = data.extract_data()
        train_ds, test_ds, class_weights= data.create_datasets(
            images, features, labels, 
            config.test_size, config.buffer_size, config.batch_size, feature_scaler_path=config.feature_scaler_path
        )
        
        trainable_model = CustomModel(train_ds, test_ds, classification_type=config.classification_type)
        
        if config.continue_train == "y":
            compiled_model = tf.keras.models.load_model(config.model_path)
        else:
            compiled_model = trainable_model.compile_model(
                config.image_shape, config.feature_shape, 
                config.learning_rate or 0.001, unfreeze_from=config.unfreeze_from, backbone=config.backbone
            )
        
        # Setup callbacks
        callbacks = _setup_callbacks(config, trainable_model)
        max_epochs = config.max_epochs or 20
        
        history = trainable_model.train_model(class_weights=class_weights,
            model=compiled_model, epochs=max_epochs,
            callbacks = callbacks,
            history_file=config.save_history_file,
            save_model_file=config.save_model_file
        )
        
        # Plot training metrics
        trainable_model.plot_metric(
            "Loss vs. Val Loss", 
            history.history['loss'], history.history['val_loss'],
            'loss', 'val_loss', 'epochs', 'loss'
        )
        trainable_model.plot_metric(
            "Accuracy vs. Val Accuracy", 
            history.history['accuracy'], history.history['val_accuracy'],
            'accuracy', 'val_accuracy', 'epochs', 'accuracy'
        )
        
    except Exception as e:
        print(f"Error during CNN training: {e}")
        raise


def _train_mlp(config) -> None:
    """
    Train an MLP model for tension prediction.

    Args:
        config: Configuration object containing training parameters
    """
    if tf is None:
        raise ImportError("TensorFlow is required for MLP training")
        
    try:
        data = MLPDataCollector(config.csv_path, config.img_folder)
        images, features, labels = data.extract_data()
        train_ds, test_ds = data.create_datasets(
            images, features, labels, 
            config.test_size, config.buffer_size, config.batch_size,
            feature_scaler_path=config.feature_scaler_path,
            tension_scaler_path=config.label_scaler_path
        )
        
        trainable_model = BuildMLPModel(config.model_path, train_ds, test_ds)
        compiled_model = trainable_model.compile_model(config.feature_shape)
        
        # Setup callbacks
        callbacks = _setup_callbacks(config, trainable_model)
        max_epochs = config.max_epochs or 20
        
        history = trainable_model.train_model(
            compiled_model, epochs=max_epochs,
            callbacks=callbacks,
            history_file=config.save_history_file,
            save_model_file=config.save_model_file
        )
        
        # Plot training metrics
        trainable_model.plot_metric(
            "Loss vs. Val Loss", 
            history.history['loss'], history.history['val_loss'],
            'loss', 'val_loss', 'epochs', 'loss'
        )
        trainable_model.plot_metric(
            "MAE vs. Val MAE", 
            history.history['mae'], history.history['val_mae'],
            'mae', 'val_mae', 'epochs', 'mae'
        )
        
    except Exception as e:
        print(f"Error during MLP training: {e}")
        raise


def _train_kfold_cnn(config) -> None:
    """
    Train CNN model using k-fold cross validation.

    Args:
        config: Configuration object containing training parameters
    """
    try:
        data = DataCollector(config.csv_path, config.img_folder, backbone=config.backbone)
        images, features, labels = data.extract_data()
        datasets = data.create_kfold_datasets(
            images, features, labels, 
            config.buffer_size, config.batch_size
        )
        
        k_models, kfold_histories = CustomModel.train_kfold(
            datasets, config.image_shape, config.feature_shape, 
            config.learning_rate or 0.001, history_file=config.save_history_file,
            save_model_file=config.save_model_file
        )
        
        CustomModel.get_averages_from_kfold(kfold_histories)
        
    except Exception as e:
        print(f"Error during k-fold CNN training: {e}")
        raise


def _train_kfold_mlp(config) -> None:
    """
    Train MLP model using k-fold cross validation.

    Args:
        config: Configuration object containing training parameters
    """
    try:
        data = MLPDataCollector(config.csv_path, config.img_folder, backbone=config.backbone)
        images, features, labels = data.extract_data()
        datasets, scaler = data.create_kfold_datasets(
            images, features, labels, 
            config.buffer_size, config.batch_size
        )
        
        k_models, kfold_histories = BuildMLPModel.train_kfold_mlp(
            datasets, config.model_path, config.feature_shape, 
            config.learning_rate or 0.001, history_file=config.save_history_file,
            save_model_file=config.save_model_file
        )
        
        BuildMLPModel.get_averages_from_kfold(kfold_histories, scaler)
        
    except Exception as e:
        print(f"Error during k-fold MLP training: {e}")
        raise


def _run_search_helper(config, tuner, train_ds, test_ds, best_params_path=None) -> None:
    """
    Helper function for running hyperparameter search.

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
        if best_params_path != None:
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
        raise


def _cnn_hyperparameter(config) -> None:
    """
    Perform hyperparameter search for CNN model.

    Args:
        config: Configuration object containing training parameters
    """
    try:
        data = DataCollector(config.csv_path, config.img_folder, backbone=config.backbone)
        images, features, labels = data.extract_data()
        train_ds, test_ds = data.create_datasets(
            images, features, labels, 
            config.test_size, config.buffer_size, config.batch_size,
            feature_scaler_path=config.feature_scaler_path
        )
        
        max_epochs = config.max_epochs or 20
        
        if config.unfreeze_from is not None:
            tuner = HyperParameterTuning(
                config.image_shape, config.feature_shape, 
                max_epochs=max_epochs, project_name=config.project_name,
                directory=config.tuner_directory, unfreeze_from=config.unfreeze_from, backbone=config.backbone
            )
        else:
            tuner = HyperParameterTuning(
                config.image_shape, config.feature_shape, 
                max_epochs=max_epochs, project_name=config.project_name,
                directory=config.tuner_directory, backbone=config.backbone
            )
        
        run_search_helper(config, tuner, train_ds, test_ds)
            
    except Exception as e:
        print(f"Error during CNN hyperparameter tuning: {e}")
        raise


def _mlp_hyperparameter(config) -> None:
    """
    Perform hyperparameter search for MLP model.

    Args:
        config: Configuration object containing training parameters
    """
    try:
        data = MLPDataCollector(config.csv_path, config.img_folder)
        images, features, labels = data.extract_data()
        train_ds, test_ds = data.create_datasets(
            images, features, labels, 
            config.test_size, config.buffer_size, config.batch_size,
            feature_scaler_path=config.feature_scaler_path,
            tension_scaler_path=config.label_scaler_path
        )
        
        max_epochs = config.max_epochs or 20
        tuner = MLPHyperparameterTuning(
            config.model_path, max_epochs=max_epochs,
            project_name=config.project_name, directory=config.tuner_directory
        )
        
        run_search_helper(config, tuner, train_ds, test_ds)
            
    except Exception as e:
        print(f"Error during MLP hyperparameter tuning: {e}")
        raise


def _test_cnn(config) -> None:
    import traceback
    """
    Test CNN model performance.

    Args:
        config: Configuration object containing test parameters
    """
    try:
        tester = TestPredictions(
            config.model_path, config.csv_path, 
            config.feature_scaler_path, config.img_folder, image_only=False, backbone=config.backbone 
        )
        true_labels, pred_labels, predictions = tester.gather_predictions()
        
        if true_labels is not None:
            tester.display_confusion_matrix(true_labels, pred_labels)
            tester.display_classification_report(
                true_labels, pred_labels, config.classification_path
            )
        else:
            print("No predictions generated - check data paths")
            
    except Exception as e:
        print(f"Error during CNN testing: {e}")
        traceback.print_exc()
        raise


def _test_mlp(config) -> None:
    """
    Test MLP model performance.

    Args:
        config: Configuration object containing test parameters
    """
    try:
        predictor = TensionPredictor(
            config.model_path, config.img_folder, config.img_path,
            config.label_scaler_path, config.feature_scaler_path
        )
        
        if config.test_features is not None:
            prediction = predictor.PredictTension(config.test_features)
            print(f"Predicted tension: {prediction}")
        else:
            print("No test features provided")
            
    except Exception as e:
        print(f"Error during MLP testing: {e}")
        raise


def _grad_cam(config) -> None:
    """
    Generate GradCAM visualization.

    Args:
        config: Configuration object containing GradCAM parameters
    """
    try:
        if config.img_path and config.test_features:
            gradcam_driver(
                config.model_path, config.img_path, config.test_features,
                backbone_name=config.backbone_name, 
                class_index=4,
                conv_layer_name='conv5_block3_out',
                heatmap_file="heatmap_6_25.png"
            )
        else:
            print("Missing image path or test features for GradCAM")
            
    except Exception as e:
        print(f"Error during GradCAM generation: {e}")
        raise


def _image_only(config) -> None:
    """
    Train image-only model (no parameter features).

    Args:
        config: Configuration object containing training parameters
    """
    import traceback
    if tf is None:
        raise ImportError("TensorFlow is required for image-only training")
        
    try:
        data = DataCollector(config.csv_path, config.img_folder, classification_type=config.classification_type, backbone=config.backbone, set_mask=config.set_mask, encoder_path=config.encoder_path)
        images, features, labels = data.extract_data()
        train_ds, test_ds, class_weights = data.create_datasets(
            images, features, labels, 
            config.test_size, config.buffer_size, config.batch_size
        )
        
        # Convert to image-only datasets
        train_ds = data.image_only_dataset(train_ds)
        test_ds = data.image_only_dataset(test_ds)
        
        trainable_model = CustomModel(train_ds, test_ds, classification_type=data.classification_type)
        if config.continue_train == "y":
            compiled_model = tf.keras.models.load_model(config.model_path)
        else:
            compiled_model = trainable_model.compile_image_only_model(
                config.image_shape, config.learning_rate or 0.001, backbone=config.backbone, dropout1_rate=config.dropout1_rate,
                dense_units=config.dense_units, dropout2_rate=config.dropout2_rate, l2_factor=config.l2_factor, num_classes=config.num_classes,
                unfreeze_from=config.unfreeze_from
            )
        # Setup callbacks
        callbacks = _setup_callbacks(config, trainable_model)
        max_epochs = config.max_epochs or 20
        
        history = trainable_model.train_model(class_weights,
            compiled_model, epochs=max_epochs,
            callbacks=callbacks,
            history_file=config.save_history_file,
            save_model_file=config.save_model_file
        )
        
        # Plot training metrics
        trainable_model.plot_metric(
            "Loss vs. Val Loss", 
            history.history['loss'], history.history['val_loss'],
            'loss', 'val_loss', 'epochs', 'loss'
        )
        trainable_model.plot_metric(
            "Accuracy vs. Val Accuracy", 
            history.history['accuracy'], history.history['val_accuracy'],
            'accuracy', 'val_accuracy', 'epochs', 'accuracy'
        )
    
    except Exception as e:
        print(f"Error during image-only training: {e}")
        traceback.print_exc()
        raise

def _test_image_only(config) -> None:
    """
    Test CNN model performance on only images.

    Args:
        config: Configuration object containing test parameters
    """
    import traceback
    try:
        tester = TestPredictions(
            model_path=config.model_path, csv_path=config.csv_path, 
             img_folder=config.img_folder, scalar_path=None, image_only=True, backbone=config.backbone,
            encoder_path=config.encoder_path, classification_type=config.classification_type
        )
        true_labels, pred_labels, predictions = tester.gather_predictions()
        
        if true_labels is not None:
            tester.display_confusion_matrix(true_labels, pred_labels)
            tester.display_classification_report(
                true_labels, pred_labels, config.classification_path
            )
        else:
            print("No predictions generated - check data paths")
            
    except Exception as e:
        print(f"Error during CNN testing: {e}")
        traceback.print_exc()
        raise


def _image_hyperparameter(config) -> None:
    """
    Perform hyperparameter search for image-only model

    Args:
        config: Configuration object containing training parameters
    """
    import traceback
    try:
        data = DataCollector(config.csv_path, config.img_folder, backbone=config.backbone, set_mask=config.set_mask, classification_type=config.classification_type)
        images, features, labels = data.extract_data()
        train_ds, test_ds, class_weights = data.create_datasets(
            images, features, labels, 
            config.test_size, config.buffer_size, config.batch_size
        )
        
        # Convert to image-only datasets
        train_ds = data.image_only_dataset(train_ds)
        test_ds = data.image_only_dataset(test_ds)
        
        max_epochs = config.max_epochs or 20
        tuner = ImageHyperparameterTuning(
            config.image_shape, max_epochs=max_epochs,
            project_name=config.project_name, directory=config.tuner_directory, backbone=config.backbone,
            num_classes=config.num_classes, class_weights=class_weights
        )
        
        run_search_helper(config, tuner, train_ds, test_ds, best_params_path=config.best_tuner_params)
    
    except Exception as e:
        print(f"Error during image hyperparameter tuning: {e}")
        traceback.print_exc()
        raise

def _custom_model(config) -> None:
    try:
        data = DataCollector(config.csv_path, config.img_folder)
        train_ds, test_ds = data.create_custom_dataset(config.image_shape, config.test_size, config.buffer_size, config.batch_size)
        trainable_model = CustomModel(train_ds, test_ds)

        compiled_model = trainable_model.compile_custom_model(config.image_shape, config.learning_rate)

        callbacks = _setup_callbacks(config, trainable_model)
        max_epochs = config.max_epochs or 20
        
        history = trainable_model.train_model(
            compiled_model, epochs=max_epochs,
            callbacks=callbacks,
            history_file=config.save_history_file,
            save_model_file=config.save_model_file
        )
        
        # Plot training metrics
        trainable_model.plot_metric(
            "Loss vs. Val Loss", 
            history.history['loss'], history.history['val_loss'],
            'loss', 'val_loss', 'epochs', 'loss'
        )
        trainable_model.plot_metric(
            "Accuracy vs. Val Accuracy", 
            history.history['accuracy'], history.history['val_accuracy'],
            'accuracy', 'val_accuracy', 'epochs', 'accuracy'
        )
    except Exception as e:
        print(f"Error during custom training: {e}")
        raise

def choices(mode: str, config) -> None:
    """
    Route to appropriate function based on mode.

    Args:
        mode: Operation mode
        config: Configuration object
    """
    mode_functions = {
        'train_cnn': _train_cnn,
        'train_mlp': _train_mlp,
        'cnn_hyperparameter': _cnn_hyperparameter,
        'mlp_hyperparameter': _mlp_hyperparameter,
        'test_cnn': _test_cnn,
        'test_mlp': _test_mlp,
        'train_kfold_cnn': _train_kfold_cnn,
        'train_kfold_mlp': _train_kfold_mlp,
        'grad_cam': _grad_cam,
        'train_image_only': _image_only,
        'image_hyperparameter': _image_hyperparameter,
        'test_image_only': _test_image_only,
        'custom_model': _custom_model
    }
    
    if mode in mode_functions:
        mode_functions[mode](config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the CLI application.

    Args:
        args: Command line arguments (optional)
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description='Fiber Cleave Processing CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cleave-app --file_path config.json
  cleave-app --file_path /path/to/config.json
        """
    )
    
    parser.add_argument(
        '--file_path', 
        type=str, 
        required=True,
        help='Path to JSON configuration file'
    )
    
    parsed_args = parser.parse_args(args)
    
    try:
        config = load_config(parsed_args.file_path)  # <-- use new factory
        choices(config.mode, config)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())  