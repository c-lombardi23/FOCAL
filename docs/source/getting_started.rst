Getting Started
===============

This quickstart will have you training your first model in only a few steps.

1. **Clone & install**  
   To clone and install in “editable” mode:

   .. code-block:: bash

      git clone https://github.com/c-lombardi23/ImageProcessing.git
      cd ImageProcessingClone
      pip install -e ".[dev]"

2. **Select a mode**  
   You drive the CLI by setting the `mode` field in your config:

   2. **Select a mode**  
   You drive the CLI by setting the `mode` field in your config:

   ``train_cnn``
       Train the CNN model using both images and numerical features
       
       - ``bad_good``: Train on binary classification
       - ``multiclass``: Train on multiple classes

   ``train_mlp``
       Train the MLP regression model (uses a pre-trained CNN)

   ``train_image_only``
       Train a CNN on images only

   ``train_kfold_cnn``
       CNN training with k-fold cross-validation

   ``train_kfold_mlp``
       MLP training with k-fold cross-validation

   ``test_cnn``
       Evaluate a trained CNN model

   ``test_mlp``
       Predict tension with a trained MLP

   ``test_image_only``
       Evaluate an image-only CNN

   ``image_hyperparameter``
       Tune hyperparameters for image-only CNN

   ``cnn_hyperparameter``
       Tune hyperparameters for combined CNN+features

   ``mlp_hyperparameter``
       Tune hyperparameters for MLP

   ``grad_cam``
       Generate Grad-CAM heatmaps

   ``custom_model``
       Train a small custom CNN from scratch

   ``train_xgboost``
       Train the XGBoost regression model

   ``test_xgboost``
       Test the XGBoost regression model


3. **Data Preparation**
 Your data should be organized as follows:

   .. code-block:: text

      your_project/
      ├── images/
      │   ├── image1.png
      │   ├── image2.png
      │   └── ...
      ├── data.csv
      └── config_files/
         └── your_config.json

   **CSV Format:** Your CSV should contain columns for image filenames and corresponding features/labels.
   Required columns are CleaveAngle, CleaveTension, ScribeDiameter, Misting, Hackle, Tearing

   **Image Requirements:**
   * Supported formats: PNG, JPG, JPEG
   * Recommended size: 224x224 pixels (will be resized automatically)
   * Images should be in a single folder

4. **Write your config file**  
   In the project root, open or create a JSON under `config_files/`, e.g. `config_files/train_cnn.json`:

   .. code-block:: json

      {
      "csv_path": "your_path_to_csv.csv",
      "img_folder": "image_folder_path",
      "feature_scaler_path": "/path_to/feature_scaler/pkl",
      "classification_path": "/path_to/classification_report.csv",

      "image_shape": [224, 224, 3],
      "feature_shape": [6],

      "mode": "train_cnn",
      "cnn_mode": "bad_good",
      "backbone": "efficientnet",
      "backbone_name": "resnet",
      "model_path": "/path_to/save_model.keras",
      "num_classes": 1,
      "classification_type": "binary",


      "learning_rate": 0.01,
      "batch_size": 16,
      "buffer_size": 40,
      "test_size": 0.25,
      "max_epochs": 50,
      "objective": "val_accuracy",
      "tension_threshold": 190,

      "brightness": 0.1,
      "height": 0.0,
      "width": 0.0,
      "contrast": 0.0,
      "rotation": 0.05,

      "dropout1": 0.0,
      "dropout2": 0.4,
      "dropout3": 0.4,
      "dense1": 64,
      "dense2": 32,
      

      "early_stopping": "n",
      "patience": 5,
      "monitor": "val_loss",
      "method": "min",
   
      "set_mask": "y"
      }

5. **Run the CLI**  

   .. code-block:: bash

      cleave-app --file_path config_files/train_cnn.json

All results (training progress, plots, reports) will print to the console and be saved wherever you pointed your `save_model_file`, `save_history_file`, etc.

6. **Common Issues:**

* **ModuleNotFoundError:** Make sure you installed in editable mode with ``pip install -e ".[dev]"``
* **Memory errors:** Reduce ``batch_size`` in your config file
* **File not found:** Check that all paths in your config use forward slashes or double backslashes

**Getting Help:**
* Check the logs for detailed error messages
* Verify your config file syntax with a JSON validator
* Make sure your CSV and image paths are correct