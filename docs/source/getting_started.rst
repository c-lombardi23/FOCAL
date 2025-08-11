.. _getting-started:

Getting Started
===============

Welcome to FOCAL (Fiber Optic Cleave Analyzer and Learner)! This guide provides a hands-on walkthrough to get you from installation to training your first classification model in just a few minutes.

.. note::
   This guide focuses on one common task to get you started quickly. For a full list of all configuration options and modes, please see the :doc:`Configuration <configuration>` page.

Prerequisites
-------------

*   Python 3.10+
*   At least 8GB RAM
*   Git

Step 1: Clone and Install
-------------------------

See the :doc:`Installation <installation>` to install the package.

Step 2: Prepare Your Data
-------------------------

The application expects your data to be organized in a specific way.

.. code-block:: text

   your_project/
   ├── images/
   │   ├── image1.png
   │   └── ...
   ├── data.csv
   └── config_files/
       └── your_config.json

- **`images/`**: A folder containing all your PNG or JPG images.
- **`data.csv`**: A CSV file containing image filenames and their corresponding features. It **must** include an ``ImagePath`` column with the relative path to the image file (e.g., `image1.png`), in addition to the feature columns: ``CleaveAngle``, ``CleaveTension``, ``ScribeDiameter``, ``Misting``, ``Hackle``, ``Tearing``.

Step 3: Create a Minimal Configuration
---------------------------------------

To begin, we will train a simple binary classification model. You only need to define a few essential settings; the application will use sensible defaults for the rest.

Create a new file at `config_files/train_cnn.json` and add the following:

.. code-block:: json

   {
       "mode": "train_cnn",
       "cnn_mode": "bad_good",

       "csv_path": "data.csv",
       "img_folder": "images/",
       "model_path": "my_first_model.keras",

       "batch_size": 16,
       "max_epochs": 10,
       "num_classes": 1,
       "classification_type": "binary",
       "train_p": 0.8,
       "test_p": 1.0,

       "diameter_threshold": 0.15,
       "angle_threshold": 0.46,

       "image_shape": [224, 224, 3],
       "feature_shape": [6],

       "dropout1": 0.0,
       "dropout2": 0.2,
       "dropout3": 0.4,
       "dense1": 62,
       "dense2": 32
   }

.. tip::
   This minimal config is all you need for a first run. For a detailed explanation of every available parameter, see the :doc:`Configuration <configuration>` reference.

Step 4: Run the Training
------------------------

You are now ready to run the command-line interface. Point it to the configuration file you just created.

.. code-block:: bash

   focal --file_path config_files/train_cnn.json

You will see training progress in the console. When it's finished, you will find `my_first_model.keras` in your project root.

Step 5: Track Your Run with MLflow
----------------------------------

This project is integrated with MLflow for experiment tracking. After your training run completes, you will see a new `mlruns` directory in your project folder.

To visualize the results, launch the MLflow UI:

.. code-block:: bash

   mlflow ui

This command reads the data from the `mlruns` directory and serves a web interface. Go to `http://localhost:5000` in your browser to view your run's parameters, metrics, and saved model artifacts.


Step 6: What's Next?
--------------------

Congratulations! You have successfully trained and saved your first model.

Now that you understand the basic workflow, you can explore more advanced features:

*   **Predict with Your Model:** Use the ``test_cnn`` mode to evaluate your saved model on new data.
*   **Try Other Models:** Experiment with ``train_mlp`` or ``train_xgboost`` for tension prediction.
*   **See All Options:** Dive into the :doc:`Configuration <configuration>` page to see all available modes and settings for fine-tuning your models.

Troubleshooting
---------------

- **ModuleNotFoundError:** Ensure you ran `pip install -e ".[dev]"`.
- **Memory errors:** Reduce the `batch_size` in your config file.
- **File not found:** Paths in your config file are relative to your project root. Ensure they are correct.