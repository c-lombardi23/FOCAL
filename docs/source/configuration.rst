.. _configuration:

Configuration Reference
=======================

The CLI is driven by a JSON configuration file. The ``mode`` field determines which operation to perform and which set of parameters are available.

This page documents all available modes and their specific configuration parameters. Since many configurations share common settings, they are grouped below.

Common Parameters
-----------------

These parameters are available in most training and testing modes.

Base Parameters
~~~~~~~~~~~~~~~

These settings form the foundation for almost every mode.

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 15 40

   * - Parameter
     - Type
     - Required?
     - Default
     - Description
   * - ``mode``
     - string
     - Yes
     - -
     - The operation to perform (e.g., ``train_cnn``, ``test_mlp``).
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
     - Set to ``"y"`` to apply a background mask to the images.

Common Training Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

These settings are available in all training modes (e.g., ``train_cnn``, ``train_mlp``, ``train_xgboost``).

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 15 40

   * - Parameter
     - Type
     - Required?
     - Default
     - Description
   * - ``model_path``
     - string (path)
     - Yes*
     - -
     - Path to save the final trained model file (e.g., ``my_model.keras``). *Functionally required for most modes.* 
   * - ``batch_size``
     - integer
     - No
     - `8`
     - Number of samples per gradient update.
   * - ``max_epochs``
     - integer
     - Yes*
     - -
     - Maximum number of epochs to train for. *Functionally required.*
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

Early Stopping Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

Enable or disable early stopping to prevent overfitting. Available in all training modes.

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 15 40

   * - Parameter
     - Type
     - Required?
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
     - Number of epochs with no improvement after which training will be stopped.
   * - ``monitor``
     - string
     - No
     - `"val_accuracy"`
     - Metric to be monitored (e.g., ``val_loss``).
   * - ``method``
     - string
     - No
     - `"max"`
     - Direction of improvement. Use ``max`` for accuracy, ``min`` for loss.


Model Checkpoint Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save the best version of your model during training. Available in all training modes.

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 15 40

   * - Parameter
     - Type
     - Required?
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
     - Path to save the best model checkpoint. *Required if ``checkpoints="y"``.*
   * - ``monitor``
     - string
     - No
     - `"val_accuracy"`
     - Metric to be monitored for saving the best model.
   * - ``method``
     - string
     - No
     - `"max"`
     - Direction of improvement (``max`` for accuracy, ``min`` for loss).

---

Mode-Specific Parameters
------------------------

This section details parameters unique to each primary mode.

.. note::
   The modes ``train_kfold_cnn`` and ``cnn_hyperparameter`` use the same configuration as ``train_cnn``. Similarly, ``train_kfold_mlp`` and ``mlp_hyperparameter`` use the ``train_mlp`` config.

train_cnn
~~~~~~~~~

Trains a Convolutional Neural Network on a combination of images and tabular features.

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 15 35

   * - Parameter
     - Type
     - Required?
     - Default
     - Description
   * - ``cnn_mode``
     - string
     - Yes
     - -
     - The classification task. Can be ``bad_good`` or ``multiclass``.
   * - ``num_classes``
     - integer
     - Yes
     - -
     - Number of output classes (e.g., `1` for binary, `5` for multiclass).
   * - ``feature_shape``
     - list[int]
     - Yes
     - -
     - Must be ``[6]`` for this mode.
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
     - The pre-trained CNN backbone to use (e.g., ``resnet``, ``mobilenet``).
   * - ``tension_threshold``
     - integer
     - No
     - `190`
     - Threshold for binary tension classification.
   * - ``unfreeze_from``
     - integer
     - No
     - `null`
     - Layer index from which to unfreeze the backbone for fine-tuning.
   * - ``reduce_lr``
     - float
     - No
     - `null`
     - Factor by which to reduce learning rate on plateau (e.g. `0.2`).
   * - ``reduce_lr_patience``
     - integer
     - No
     - `null`
     - Number of epochs to wait before reducing LR.

train_mlp
~~~~~~~~~

Trains a Multi-Layer Perceptron (MLP) regression model on tabular features.

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 15 35

   * - Parameter
     - Type
     - Required?
     - Default
     - Description
   * - ``img_path``
     - string (path)
     - Yes
     - -
     - Path to an image required by the MLP's data processing pipeline.
   * - ``feature_shape``
     - list[int]
     - Yes
     - -
     - Must be ``[5]`` for this mode.

train_xgboost
~~~~~~~~~~~~~

Trains an XGBoost regression model on tabular features.

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 15 35

   * - Parameter
     - Type
     - Required?
     - Default
     - Description
   * - ``xgb_path``
     - string (path)
     - Yes*
     - `null`
     - Path to save the trained XGBoost model. *Functionally required.*
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
   * - ``random_state``
     - integer
     - No
     - `42`
     - Random seed for reproducibility.

grad_cam
~~~~~~~~

Generates a Grad-CAM heatmap to visualize model focus.

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 15 35

   * - Parameter
     - Type
     - Required?
     - Default
     - Description
   * - ``model_path``
     - string (path)
     - Yes*
     - `null`
     - Path to the trained CNN model.
   * - ``img_path``
     - string (path)
     - Yes*
     - `null`
     - Path to the specific image for visualization.
   * - ``test_features``
     - list[float]
     - Yes*
     - `null`
     - Tabular features corresponding to the test image.
   * - ``backbone_name``
     - string
     - Yes*
     - `null`
     - Name of the backbone used in the saved model.
   * - ``conv_layer_name``
     - string
     - Yes*
     - `null`
     - Name of the target convolutional layer for Grad-CAM.
   * - ``heatmap_file``
     - string (path)
     - No
     - `null`
     - Path to save the output heatmap image.