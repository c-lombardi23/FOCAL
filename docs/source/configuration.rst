.. _configuration:

Configuration Reference
=======================

The CLI is driven by a single JSON configuration file. The ``mode`` field is mandatory and determines which operation to perform and which set of parameters are available. This page documents all available modes and their specific configurations.

.. note::
   An asterisk (Yes\*) in the "Required" column indicates that a parameter is conditionally required. For example, `checkpoint_filepath` is only required if `checkpoints` is set to `"y"`. The Pydantic models in ``config_schema.py`` are the ultimate source of truth for all validations.


Example Configuration
---------------------
Here is an example of a ``config.json`` for the ``train_cnn`` mode:

.. code-block:: json

   {
     "mode": "train_cnn",
     "csv_path": "data/cleaves.csv",
     "img_folder": "data/images/",
     "image_shape": [224, 224, 3],
     "feature_shape": [5],
     "save_model_file": "models/my_cnn_model.keras",
     "max_epochs": 15,
     "cnn_mode": "bad_good",
     "classification_type": "binary",
     "num_classes": 1,

    "learning_rate": 0.005,
    "batch_size": 16,
    "buffer_size": 40,
    "test_size": 0.25,
    "max_epochs": 2,
    "objective": "val_accuracy",

    "brightness": 0.3,
    "height": 0.2,
    "width": 0.6,
    "contrast": 0.7,
    "rotation": 0.45,

     "angle_threshold": 0.5,
     "diameter_threshold": 125.0,
     "dense1": 32,
     "dense2": 16,
     "dropout1": 0.2,
     "dropout2": 0.2,
     "dropout3": 0.2,
     "checkpoints": "y",
     "checkpoint_filepath": "models/checkpoints/best_cnn.keras"
   }

---

Common Parameters
-----------------

Base Parameters
~~~~~~~~~~~~~~~

These settings form the foundation for almost every mode.

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 15 40

   * - Parameter
     - Type
     - Required
     - Default
     - Description
   * - ``mode``
     - string
     - Yes
     - -
     - The operation to perform. Must be one of the documented modes.
   * - ``csv_path``
     - string (path)
     - Yes
     - -
     - Path to the input data CSV file.
   * - ``img_folder``
     - string (path)
     - Yes
     - -
     - Path to the folder containing all images.
   * - ``image_shape``
     - list[int]
     - Yes
     - -
     - Dimensions of the input image, e.g., ``[224, 224, 3]``.
   * - ``feature_shape``
     - list[int]
     - No
     - `null`
     - Dimensions of the tabular feature vector. Required by some modes.
   * - ``set_mask``
     - string
     - No
     - `null`
     - Set to ``"y"`` to apply a circular background mask to the images.

Common Training Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

These settings are available in all training modes (e.g., ``train_cnn``, ``train_mlp``, ``train_image_only``, ``train_xgboost``).

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 15 40

   * - Parameter
     - Type
     - Required
     - Default
     - Description
   * - ``save_model_file``
     - string (path)
     - No
     - `null`
     - Path to save the final trained model file. Recommended for all training modes.
   * - ``save_history_file``
     - string (path)
     - No
     - `null`
     - Path to save the training history to a CSV file.
   * - ``batch_size``
     - integer
     - No
     - `8`
     - Number of samples per gradient update.
   * - ``max_epochs``
     - integer
     - No
     - `null`
     - Maximum number of epochs to train for.
   * - ``learning_rate``
     - float
     - No
     - `0.001`
     - The learning rate for the optimizer.
   * - ``test_size``
     - float
     - No
     - `0.2`
     - Proportion of the dataset to use for the validation split (0.0 to 1.0).
   * - ``feature_scaler_path``
     - string (path)
     - No
     - `null`
     - Path to save the trained feature scaler.
   * - ``label_scaler_path``
     - string (path)
     - No
     - `null`
     - Path to save the trained label scaler (for regression).
   * - ``encoder_path``
     - string (path)
     - No
     - `null`
     - Path to save the trained label encoder (for classification).
   * - ``brightness``, ``rotation``, ``height``, ``width``, ``contrast``
     - float
     - No
     - `0.0`
     - Parameters for image data augmentation.

Common Callback Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

These settings control `keras.callbacks` and are available in all TensorFlow-based training modes.

Early Stopping
""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 15 40

   * - Parameter
     - Type
     - Required
     - Default
     - Description
   * - ``early_stopping``
     - string
     - No
     - `"n"`
     - Set to ``"y"`` to enable.
   * - ``patience``
     - integer
     - No
     - `3`
     - Epochs with no improvement before stopping training.
   * - ``monitor``
     - string
     - No
     - `"val_accuracy"`
     - Metric to monitor (e.g., ``val_loss``).
   * - ``method``
     - string
     - No
     - `"max"`
     - Direction of improvement. Use ``max`` for accuracy, ``min`` for loss.

Model Checkpointing
"""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 15 40

   * - Parameter
     - Type
     - Required
     - Default
     - Description
   * - ``checkpoints``
     - string
     - No
     - `"n"`
     - Set to ``"y"`` to enable model checkpointing.
   * - ``checkpoint_filepath``
     - string (path)
     - Yes*
     - `null`
     - Path to save the best model checkpoint. Required if `checkpoints="y"`.
   * - ``monitor``
     - string
     - No
     - `"val_accuracy"`
     - Metric to monitor for saving the best model.
   * - ``method``
     - string
     - No
     - `"max"`
     - Direction of improvement (``max`` for accuracy, ``min`` for loss).

---

Mode-Specific Parameters
------------------------

Training Modes
~~~~~~~~~~~~~~

train_cnn
"""""""""
Trains a hybrid model on a combination of images and tabular features.

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 15 35

   * - Parameter
     - Type
     - Required
     - Default
     - Description
   * - ``cnn_mode``
     - string
     - Yes
     - -
     - The classification task. Can be ``bad_good`` or ``multiclass``.
   * - ``classification_type``
     - string
     - Yes
     - `binary`
     - Must be ``binary`` or ``multiclass``.
   * - ``num_classes``
     - integer
     - Yes
     - -
     - Number of output classes (e.g., `1` for binary, `5` for multiclass).
   * - ``feature_shape``
     - list[int]
     - Yes
     - -
     - Must be ``[5]`` for this mode.
   * - ``angle_threshold``
     - float
     - Yes
     - -
     - Threshold for angle-based classification logic.
   * - ``diameter_threshold``
     - float
     - Yes
     - -
     - Threshold for diameter-based classification logic.
   * - ``train_p``
     - float
     - Yes
     - -
     - Masking probability for training features.
   * - ``test_p``
     - float
     - Yes
     - -
     - Masking probability for testing features.
   * - ``dense1``, ``dense2``
     - integer
     - Yes
     - -
     - Number of units in the two dense layers of the model head.
   * - ``dropout1``, ``dropout2``, ``dropout3``
     - float
     - Yes
     - -
     - Dropout rates for regularization.
   * - ``backbone``
     - string
     - No
     - `efficientnet`
     - The pre-trained CNN backbone (``resnet``, ``mobilenet``, ``efficientnet``).
   * - ``unfreeze_from``
     - integer
     - No
     - `null`
     - Layer index from which to unfreeze the backbone for fine-tuning.
   * - ``reduce_lr``
     - float
     - No
     - `null`
     - Factor to reduce learning rate on plateau (e.g. `0.2`).
   * - ``reduce_lr_patience``
     - integer
     - No
     - `null`
     - Epochs to wait before reducing LR.

train_mlp
"""""""""
Trains an MLP regression model using features extracted from a pre-trained CNN.

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 15 35

   * - Parameter
     - Type
     - Required
     - Default
     - Description
   * - ``model_path``
     - string (path)
     - Yes
     - -
     - Path to the **pre-trained CNN model** used for feature extraction.
   * - ``feature_shape``
     - list[int]
     - Yes
     - -
     - Must be ``[4]`` for this mode (the numerical features, excluding tension).
   * - ``angle_threshold``, ``diameter_threshold``
     - float
     - Yes
     - -
     - Thresholds required for the data processing pipeline.
   * - ``dense1``, ``dense2``, ``dropout1``, etc.
     - float/int
     - Yes
     - -
     - Architecture parameters for the MLP model.
   * - ``reduce_lr``
     - float
     - No
     - `null`
     - Factor to reduce learning rate on plateau (e.g. `0.2`).
   * - ``reduce_lr_patience``
     - integer
     - No
     - `null`
     - Epochs to wait before reducing LR.

train_image_only
""""""""""""""""
Trains a classification model using only images as input.

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 15 35

   * - Parameter
     - Type
     - Required
     - Default
     - Description
   * - ``backbone``
     - string
     - Yes
     - -
     - The pre-trained CNN backbone to use.
   * - ``classification_type``
     - string
     - Yes
     - -
     - Must be ``binary`` or ``multiclass``.
   * - ``num_classes``
     - integer
     - Yes
     - -
     - Number of output classes.
   * - ``angle_threshold``, ``diameter_threshold``
     - float
     - Yes
     - -
     - Thresholds for defining labels.
   * - ``dense1``, ``dropout1``, ``dropout2``, ``l2_factor``
     - float/int
     - No
     - `various`
     - Architecture parameters for the model head.

train_xgboost
"""""""""""""
Trains an XGBoost regression model.

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 15 35

   * - Parameter
     - Type
     - Required
     - Default
     - Description
   * - ``xgb_path``
     - string (path)
     - No
     - `null`
     - Path to save the trained XGBoost model (`.pkl`). Recommended.
   * - ``model_path``
     - string (path)
     - Yes
     - -
     - Path to the pre-trained CNN used for feature extraction.
   * - ``angle_threshold``, ``diameter_threshold``
     - float
     - Yes
     - -
     - Thresholds for data processing.
   * - ``error_type``
     - string
     - Yes
     - -
     - The XGBoost objective function (e.g., `reg:squarederror`).
   * - ``n_estimators``
     - integer
     - No
     - `200`
     - Number of gradient boosted trees.
   * - ``max_depth``
     - integer
     - No
     - `4`
     - Maximum tree depth for base learners.
   * - ``gamma``, ``subsample``, ``reg_lambda``
     - float
     - No
     - `various`
     - Regularization and subsampling parameters for XGBoost.

Testing & Evaluation Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~

test_cnn & test_image_only
""""""""""""""""""""""""""
Tests a saved image-based classifier and generates evaluation reports.

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 15 35

   * - Parameter
     - Type
     - Required
     - Default
     - Description
   * - ``model_path``
     - string (path)
     - Yes
     - -
     - Path to the trained classifier model (`.keras`).
   * - ``angle_threshold``, ``diameter_threshold``
     - float
     - Yes
     - -
     - Thresholds used to generate the ground-truth labels for comparison.
   * - ``classification_path``
     - string (path)
     - No
     - `null`
     - Path to save the output CSV classification report.
   * - ``classification_threshold``
     - float
     - No
     - `0.5`
     - The probability threshold for binary classification.
   * - ``feature_scaler_path``
     - string (path)
     - No
     - `null`
     - Required for ``test_cnn`` if the model used scaled features.

test_mlp & test_xgboost
"""""""""""""""""""""""
Tests a saved regression model and generates a performance report.

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 15 35

   * - Parameter
     - Type
     - Required
     - Default
     - Description
   * - ``model_path``
     - string (path)
     - Yes
     - -
     - Path to the trained regressor (`.keras` for MLP) or feature extractor (`.keras` for XGBoost).
   * - ``xgb_path``
     - string (path)
     - Yes*
     - `null`
     - Required for ``test_xgboost`` mode. Path to the `.pkl` file.
   * - ``angle_threshold``, ``diameter_threshold``
     - float
     - Yes
     - -
     - Thresholds for data processing.
   * - ``label_scaler_path``
     - string (path)
     - Yes*
     - `null`
     - Path to the saved label scaler used during training. Required.

.. note::

   For the ``test_xgboost`` mode, the ``model_path`` parameter should point to the pre-trained **CNN feature extractor** model (`.keras`), not the XGBoost model itself.

Advanced Modes
~~~~~~~~~~~~~~

K-Fold Cross-Validation
"""""""""""""""""""""""
The ``train_kfold_cnn`` and ``train_kfold_mlp`` modes are used for more robust model evaluation. They accept the exact same parameters as their non-k-fold counterparts (``train_cnn`` and ``train_mlp`` respectively), with the addition of `n_splits` if you want to change the number of folds.

Hyperparameter Tuning
"""""""""""""""""""""
The ``cnn_hyperparameter``, ``mlp_hyperparameter``, and ``image_hyperparameter`` modes are used to search for the best model architecture.

- ``cnn_hyperparameter`` uses the same config as ``train_cnn``.
- ``image_hyperparameter`` uses the same config as ``train_image_only``.
- ``mlp_hyperparameter`` requires ``tuner_directory`` and ``project_name``.

Visualization (grad_cam)
""""""""""""""""""""""""
Generates a Grad-CAM heatmap to visualize which parts of an image the CNN is focusing on.

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 15 35

   * - Parameter
     - Type
     - Required
     - Default
     - Description
   * - ``model_path``
     - string (path)
     - Yes
     - -
     - Path to the trained CNN model.
   * - ``img_path``
     - string (path)
     - Yes
     - -
     - Path to the specific image for visualization.
   * - ``test_features``
     - list[float]
     - Yes*
     - `null`
     - Required if the model takes numerical inputs.
   * - ``class_index``
     - int
     - Yes
     - -
     - Index of classification problem.
   * - ``backbone``
     - string
     - No
     - `null`
     - The name of the backbone layer in the saved model (e.g., `'mobilenet'`).
   * - ``conv_layer_name``
     - string
     - No
     - `null`
     - Name of the target convolutional layer. If `null`, the last conv layer is used.
   * - ``heatmap_file``
     - string (path)
     - No
     - `null`
     - Path to save the output heatmap image.
   * - ``backbone``
     - string
     - No
     - `efficientnet`
     - Name of pre-trained backbone.
   

Reinforcement Learning
~~~~~~~~~~~~~~~~~~~~~~

train_rl & test_rl
""""""""""""""""""
Train or test an agent with reinforcement learning to predict optimal tension.

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 15 35

   * - Parameter
     - Type
     - Required
     - Default
     - Description
   * - ``csv_path``
     - string (path)
     - Yes
     - -
     - Path to the csv dataset.
   * - ``cnn_path``
     - string (path)
     - Yes
     - -
     - Path to the cnn classifier.
   * - ``img_folder``
     - string (path)
     - Yes
     - -
     - Path to the saved images.
   * - ``agent_path``
     - string (path)
     - Yes
     - -
     - Path to save (or load) trained agent.
   * - ``learning_rate``
     - float
     - Yes
     - -
     - Typical learning rate for ML.
   * - ``buffer_size``
     - int
     - Yes
     - -
     - Size of replay buffer.
   * - ``threshold``
     - float
     - Yes
     - -
     - Classification threshold.
   * - ``max_tension_change``
     - float
     - Yes
     - -
     - Maximum tension change per episode. 
   * - ``batch_size``
     - int
     - No
     - `256``
     - Batch for training.
   * - ``tau``
     - float
     - No
     - `0.1`
     - -
   * - ``learning_rate``
     - float
     - No
     - `0.0001`
     - Size of steps to take during training.
   * - ``timesteps``
     - int
     - No
     - `5000`
     - Number of training rounds.
   * - ``low_range```
     - float
     - No
     - `0.7`
     - Low percentage of tension.
   * - ``high_range``
     - float
     - No
     - `1.4`
     - High percentage of tension.  
