from .config_schema import Config
import warnings
import os

# ignore warnings from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


from .data_processing import *
from .model_pipeline import *
from .prediction_testing import *
from .hyperparameter_tuning import *

import argparse
import json


def load_file(filepath):
    '''
    Load a json file from path and calls Config function
    to parse data.

    Parameters:
    ----------------------------------

    filepath: str
        - path to json file
    
    Returns: Config.BaseModel
        - Parsed file contents
    '''
    with open(filepath, 'r') as f:
        data = json.load(f)
    return Config(**data)

def train_cnn(config):
    '''
    Logic for training CNN model from model_pipeline.

    Parameters:
    --------------------------------------

    config: Config.BaseModel
        - parsed file contents
    
    '''
    data = DataCollector(config.csv_path, config.img_folder)
    images, features, labels = data.extract_data()
    train_ds, test_ds = data.create_datasets(images, features, labels, config.test_size, config.buffer_size, config.batch_size)
    trainable_model = CustomModel(train_ds, test_ds)
    if config.continue_train == "y":
        compiled_model = tf.keras.models.load_model(config.model_path)
    else:
        compiled_model = trainable_model.compile_model(config.image_shape, config.feature_shape, config.learning_rate, unfreeze_from=config.unfreeze_from)
    if config.checkpoints == "y":
        checkpoint = trainable_model.create_checkpoints(config.checkpoint_filepath, config.monitor, config.method)
    else:
        checkpoint = None
    if config.early_stopping == "y":
        es = trainable_model.create_early_stopping(config.patience, config.method, config.monitor)
    else:
        es = None
    if config.reduce_lr != None:
        if config.reduce_lr_patience != None:
            reduce_lr = trainable_model.reduce_on_plateau(patience=config.reduce_lr_patience, factor=config.reduce_lr)
        else:
            reduce_lr = trainable_model.reduce_on_plateau(factor=config.reduce_lr)
    else:
        reduce_lr = None
    if config.max_epochs == None:
        config.max_epochs = 20
    
    history = trainable_model.train_model(compiled_model, epochs=config.max_epochs, early_stopping=es, reduce_lr=reduce_lr, checkpoints=checkpoint, history_file=config.save_history_file, model_file=config.save_model_file)
    trainable_model.plot_metric("Loss vs. Val Loss", history.history['loss'], history.history['val_loss'], 'loss', 'val_loss', 'epochs', 'loss')
    trainable_model.plot_metric("Accuracy vs. Val Accuracy", history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 'val_accuracy', 'epochs', 'accuracy')


def train_mlp(config):
    '''
    Logic for training regression model. 

    Parameters:
    ----------------------------------

    config: Config.BaseModel
        - parsed file contents
    '''
    data = MLPDataCollector(config.csv_path, config.img_folder)
    images, features, labels = data.extract_data()
    train_ds, test_ds = data.create_datasets(images, features, labels, config.test_size, config.buffer_size, config.batch_size, feature_scaler_path=config.feature_scaler_path, tension_scaler_path=config.label_scaler_path)
    trainable_model = BuildMLPModel(config.model_path, train_ds, test_ds)
    compiled_model = trainable_model.compile_model(config.feature_shape)
    if config.checkpoints == "y":
        checkpoint = trainable_model.create_checkpoints(config.checkpoint_filepath, config.monitor, config.method)
    else:
        checkpoint = None
    if config.early_stopping == "y":
        es = trainable_model.create_early_stopping(config.patience, config.method, config.monitor)
    else:
        es = None
    if config.max_epochs == None:
        config.max_epochs = 20
    history = trainable_model.train_model(compiled_model, epochs=config.max_epochs, early_stopping=es, reduce_lr=config.reduce_lr, checkpoints=checkpoint, history_file=config.save_history_file, model_file=config.save_model_file)
    trainable_model.plot_metric("Loss vs. Val Loss", history.history['loss'], history.history['val_loss'], 'loss', 'val_loss', 'epochs', 'loss')
    trainable_model.plot_metric("MAE vs. Val MAE", history.history['mae'], history.history['val_mae'], 'mae', 'val_mae', 'epochs', 'mae')

def train_kfold_cnn(config):
    '''
    Logic for training CNN model using k-fold cross validation.

    Parameters:
    -----------------------------

    config: Config.BaseModel
        - parsed file contents
    '''
    data = DataCollector(config.csv_path, config.img_folder)
    images, features, labels = data.extract_data()
    datasets = data.create_kfold_datasets(images, features, labels, config.buffer_size, config.batch_size)
    k_models, kfold_histories = CustomModel.train_kfold(datasets, config.image_shape, config.feature_shape, config.learning_rate, history_file = config.save_history_file,
                                            model_file = config.save_model_file)
    CustomModel.get_averages_from_kfold(kfold_histories)


def train_kfold_mlp(config):
    '''
    Logic for training regression model using k-fold cross validation

    Parameters:
    --------------------------------------

    config: Config.BaseModel
        - parsed file contents
    '''
    data = MLPDataCollector(config.csv_path, config.img_folder)
    images, features, labels = data.extract_data()
    datasets = data.create_kfold_datasets(images, features, labels, config.buffer_size, config.batch_size)
    k_models, kfold_histories = BuildMLPModel.train_kfold_mlp(datasets, config.model_path, config.feature_shape, config.learning_rate, history_file = config.save_history_file,
                                            model_file = config.save_model_file)
    BuildMLPModel.get_averages_from_kfold(kfold_histories)

def run_search_helper(config, tuner, train_ds, test_ds):
    '''
    Helper function for running search in keras tuner.

    Parameters:
    ----------------------------------------

    config: Config.BaseModel
        - parsed file contents
    tuner: keras_tuner.Tuner
        - instance of tuner class of Keras models
    train_ds: tf.data.Dataset
        - dataset of training elements
    test_ds: tf.data.Dataset
        - dataset of testing elements
    '''
    tuner.run_search(train_ds, test_ds)
    print(tuner.get_best_hyperparameters().values)
    pathname = config.best_model_path
    if pathname == None:
        print("Model not saved")
        exit()
    else:
        tuner.save_best_model(pathname)
        print(f"Model saved to: {pathname}")

def cnn_hyperparameter(config):
    '''
    Logic for performing hyperparameter search on CNN model.

    Parameters:
    -------------------------------------------

    config: Config.BaseModel
        - parsed file contents
    '''
    data = DataCollector(config.csv_path, config.img_folder)
    images, features, labels = data.extract_data()
    train_ds, test_ds = data.create_datasets(images, features, labels, config.test_size, config.buffer_size, config.batch_size)
    if config.max_epochs == None:
        config.max_epochs = 20
    if config.unfreeze_from != None:
        tuner = HyperParameterTuning(config.image_shape, config.feature_shape, max_epochs=config.max_epochs, project_name=config.project_name, directory=config.tuner_directory, unfreeze_from=config.unfreeze_from)
    else:
        tuner = HyperParameterTuning(config.image_shape, config.feature_shape, max_epochs=config.max_epochs, project_name=config.project_name, directory=config.tuner_directory)
    run_search_helper(config, tuner, train_ds, test_ds)
    if config.tuner_path != None:
        tuner.save_best_model(config.tuner_path)
  
def mlp_hyperparameter(config):
    '''
    Logic for performing hyperparameter search on regression model.

    Parameters:
    ------------------------------------------

    config: Config.BaseModel
        - parsed file contents
    '''
    data = MLPDataCollector(config.csv_path, config.img_folder)
    images, features, labels = data.extract_data(config.feature_scaler_path)
    train_ds, test_ds = data.create_datasets(images, features, labels, config.test_size, config.buffer_size, config.batch_size)
    if config.max_epochs == None:
        config.max_epochs = 20
    tuner = MLPHyperparameterTuning(config.image_shape, config.feature_shape, max_epochs=config.max_epochs, project_name=config.project_name, directory=config.tuner_directory)
    run_search_helper(config, tuner, train_ds, test_ds)
    if config.tuner_path != None:
        tuner.save_best_model(config.tuner_path)

def test_cnn(config):
    '''
    Logic for testing CNN model.

    Parameters:
    ------------------------------------------

    config: Config.BaseModel
        - parsed file contents
    '''
    tester = TestPredictions(config.model_path, config.csv_path, config.feature_scaler_path, config.img_folder)
    true_labels, pred_labels, predictions = tester.gather_predictions()
    tester.display_confusion_matrix(true_labels, pred_labels)
    tester.display_classification_report(true_labels, pred_labels, config.classification_path)

def test_mlp(config):
    '''
    Logic for testing regression model.

    Parameters:
    ------------------------------------------

    config: Config.BaseModel
        - parsed file contents
    '''
    test_model = tf.keras.models.load_model(config.model_path)
    tester = TensionPredictor(test_model, config.img_folder, config.img_path, config.label_scaler_path, config.feature_scaler_path)
    predicted_tension = tester.PredictTension(config.test_features)
    print(f"Predicted Tension: {predicted_tension:.0f}g")
 

def choices(mode, config):
    '''
    Call function based on mode input.

    Parameters: 
    ------------------------------------------------

    mode: str
        - training or testing mode to call
    config: Config.BaseModel
        - parsed file contents 
    '''
    if mode == "train_cnn":
        train_cnn(config)
    elif mode == "train_mlp":
        train_mlp(config)
    elif mode == "cnn_hyperparameter":
        cnn_hyperparameter(config)
    elif mode == "mlp_hyperparameter":
        mlp_hyperparameter(config)
    elif mode == "test_cnn":
        test_cnn(config)
    elif mode == "test_mlp":
        test_mlp(config)
    elif mode == "train_kfold_cnn":
        train_kfold_cnn(config)
    elif mode == "train_kfold_mlp":
        train_kfold_mlp(config)

def main(args=None):
    # parse file_path entry from CLI
    parser = argparse.ArgumentParser(description="Train Model from command line")
    parser.add_argument("--file_path", required=True)
    parsed_args = parser.parse_args(args)

    # extract file path
    filepath = parsed_args.file_path
    config = load_file(filepath)
    mode = config.mode
    # call choices function
    choices(mode, config)


if __name__ == "__main__":
    main()  