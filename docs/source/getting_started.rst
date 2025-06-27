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

   .. list-table::
      :header-rows: 0

      * - ``train_cnn``         - Train the CNN model using both images and numerical features
      * - ``train_mlp``         - Train the MLP regression model (uses a pre-trained CNN)
      * - ``train_image_only``  - Train a CNN on images only
      * - ``train_kfold_cnn``   - CNN training with k-fold cross-validation
      * - ``train_kfold_mlp``   - MLP training with k-fold cross-validation
      * - ``test_cnn``          - Evaluate a trained CNN model
      * - ``test_mlp``          - Predict tension with a trained MLP
      * - ``test_image_only``   - Evaluate an image-only CNN
      * - ``image_hyperparameter``   - Tune hyperparameters for image-only CNN
      * - ``cnn_hyperparameter``     - Tune hyperparameters for combined CNN+features
      * - ``mlp_hyperparameter``     - Tune hyperparameters for MLP
      * - ``grad_cam``                - Generate Grad-CAM heatmaps
      * - ``custom_model``            - Train a small custom CNN from scratch

3. **Write your config file**  
   In the project root, open or create a JSON under `config_files/`, e.g. `config_files/train_cnn.json`:

   .. code-block:: json

      {
        "mode": "train_cnn",
        "csv_path":  "data/cleave_metadata.csv",
        "img_folder": "data/images",
        "feature_scaler_path": "models/feature_scaler.pkl",
        "label_scaler_path":   "models/label_scaler.pkl",
        "image_shape": [224, 224, 3],
        "feature_shape": [6],
        "test_size":  0.2,
        "buffer_size": 32,
        "batch_size":  16,
        "learning_rate": 0.001,
        "max_epochs":   50,
        "early_stopping": "y",
        "patience":      5,
        "monitor":       "val_accuracy",
        "checkpoints":   "y",
        "checkpoint_filepath": "checkpoints/best_model.keras",
        "save_model_file":     "models/final_model.keras",
        "save_history_file":   "history/train_history.csv",
        "set_mask": "y"
      }

4. **Run the CLI**  

   .. code-block:: bash

      cleave-app --file_path config_files/train_cnn.json

All results (training progress, plots, reports) will print to the console and be saved wherever you pointed your `save_model_file`, `save_history_file`, etc.
