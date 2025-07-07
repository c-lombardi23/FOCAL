# Fiber Cleave Processing

![alt text](thorlabs-logo.png)


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.19+](https://img.shields.io/badge/tensorflow-2.19+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning package for fiber cleave quality classification and tension prediction using CNN and MLP models.

## 📋 Table of Contents

- [Project Description](#project-description)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Tips for Better Accuracy](#tips-for-better-accuracy)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Description

This project implements a comprehensive machine learning pipeline for analyzing fiber cleave quality using images from the THORLABS Fiber Cleave Analyzer (FCA). The system consists of three main components:

1. **CNN Classification Model**: Classifies cleave images as good or bad based on visual features alone or inclusiong of numerical features
2. **MLP Regression Model**: Predicts optimal tension parameters for producing good cleaves
3. **XGBoost Regression Model**: Predicts the change in tension needed to produce a good cleave 

The models use transfer learning with either MobileNetV2, ResNet, or EfficientNetB0 as the backbone and are optimized using Keras Tuner for hyperparameter optimization.

## ✨ Key Features

- **Transfer Learning**: Uses pre-trained models for robust feature extraction
- **Multi-Modal Input**: Combines image features with numerical parameters
- **Hyperparameter Optimization**: Automated tuning using Keras Tuner
- **Flexible Architecture**: Supports both classification and regression tasks
- **K-Fold Cross Validation**: Robust model evaluation
- **GradCAM Visualization**: Model interpretability through heatmaps
- **Command Line Interface**: Easy-to-use CLI for training and inference
- **Comprehensive Logging**: Training history and model checkpoints

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.19 or higher
- CUDA-compatible GPU (recommended for training)

### Install from PyPI

```bash
pip install FiberCleaveProcessing
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/c-lombardi23/ImageProcessing.git
cd ImageProcessing

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
cleave-app --help
```

## 🏃‍♂️ Quick Start

1. **Prepare your data**:
   - Organize cleave images in a folder
   - Create a CSV file with metadata (tension, angle, scribe diameter, misting, hackle, tearing)

2. **Create a configuration file** (see [Configuration](#configuration) section)

3. **Train a classification model**:
   ```bash
   cleave-app --file_path config.json
   ```

## ⚙️ Configuration

The application uses a JSON configuration file to specify all parameters. **Each mode uses a dedicated config class with its own required and optional fields.** The CLI automatically loads the correct config class for the selected mode.

> **Note:** Not all parameters are required for every mode. Refer to the mode-specific config class or config.schema.json file for required/optional fields. The system will validate your config and provide clear errors if required fields are missing.

### Example: `train_cnn` Configuration

```json
{
  "csv_path": "C:\\Thorlabs\\125PM_2_Categories.csv",
  "img_folder": "C:\\Thorlabs\\125PM\\",
  "feature_scaler_path": "C:\\Users\\clombardi\\Training_Runs_7_7\\Individual_Trials\\images_features1.pkl",
  "label_scaler_path": "C:\\Users\\clombardi\\Training_Runs_7_7\\Individual_Trials\\label_scaler1.pkl",
  "classification_path": "C:\\Users\\clombardi\\125pm_test1.csv",
  "img_path": "C:\\Thorlabs\\125PM\\Fiber-312Plus.png",

  "image_shape": [224, 224, 3],
  "feature_shape": [6],
  "test_features": [1.68, 190, 12.96, 0, 0, 1],

  "mode": "train_cnn",
  "cnn_mode": "bad_good",
  "backbone": "efficientnet",
  "backbone_name": "resnet",
  "model_path": "C:\\Users\\clombardi\\Training_Runs_7_7\\Individual_Trials\\image_features1.keras",
  "project_name": "Binary4",
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
  "checkpoints": "y",
  "checkpoint_filepath": "C:\\Users\\clombardi\\Training_Runs_7_7\\Individual_Trials\\images_features1_checkpoint.keras",

  "tuner_directory": "C:\\Users\\clombardi\\Training_Runs_6_27\\HyperParameterTuning1",
  "save_model_file": "C:\\Users\\clombardi\\Training_Runs_7_7\\Individual_Trials\\images_features1.keras",
  "save_history_file": "C:\\Users\\clombardi\\Training_Runs_7_7\\Individual_Trials\\images_features1_history",

  "set_mask": "y"
}
```

### Configuration Parameters

- **`mode`**: Operation mode (see table below)
- **Other fields**: See the mode-specific config class in `src/cleave_app/config_schema.py` for required/optional fields for each mode.
- **Validation**: The CLI will validate your config and provide clear errors if required fields are missing or invalid for the selected mode.

### Available Modes

| Mode | Description |
|------|-------------|
| `train_cnn` | Train CNN classification model |
| `train_mlp` | Train MLP regression model |
| `train_image_only` | Train image-only classification model |
| `train_xgboost` | Train XGBoost regressor |
| `cnn_hyperparameter` | Run hyperparameter search for CNN |
| `mlp_hyperparameter` | Run hyperparameter search for MLP |
| `image_hyperparameter` | Run hyperparameter search for image-only model |
| `test_cnn` | Test CNN model performance |
| `test_mlp` | Test MLP model performance |
| `test_image_only` | Test the CNN on only image data |
| `test_xgboost`  | Test the XGBoost regressor  |
| `train_kfold_cnn` | Train CNN with k-fold cross validation |
| `train_kfold_mlp` | Train MLP with k-fold cross validation |
| `grad_cam` | Generate GradCAM visualizations |

> **Extensibility:** To add a new mode, simply add a new config class in `config_schema.py` and update the mode-to-config mapping.

## 📖 Usage Examples

### Training a Classification Model

```bash
# Train CNN model
cleave-app --file_path config_cnn.json

# Train with hyperparameter optimization
cleave-app --file_path config_cnn_tuner.json
```

### Training a Regression Model

```bash
# Train MLP model for tension prediction
cleave-app --file_path config_mlp.json
```

### Testing Models

```bash
# Test classification model
cleave-app --file_path config_test_cnn.json

# Test regression model
cleave-app --file_path config_test_mlp.json
```

### K-Fold Cross Validation

```bash
# Train with k-fold cross validation
cleave-app --file_path config_kfold.json
```

## 🧠 Tips for Better Accuracy

If your image-only model is not achieving the desired accuracy, consider the following strategies:

- **Fine-tune the Pretrained Backbone:** Unfreeze the last 10–20 layers of the pre-trained and continue training with a low learning rate.
- **Increase Model Capacity:** Add more dense layers or increase the number of units after the global average pooling layer.
- **Tune Data Augmentation:** Use more aggressive or varied augmentations (e.g., `RandomFlip`, higher `RandomBrightness`/`RandomContrast`).
- **Regularization:** Add or increase dropout, or use L2 regularization on dense layers.
- **Learning Rate Scheduling:** Use a learning rate scheduler or `ReduceLROnPlateau` callback.
- **Handle Class Imbalance:** Use class weights or oversample minority classes if your dataset is imbalanced.
- **Hyperparameter Tuning:** Use Keras Tuner to search for the best architecture and training parameters.
- **Image Preprocessing:** Ensure images are normalized to the range expected by the backbone (done in code already).
- **Train Longer with Early Stopping:** Allow more epochs and use early stopping to avoid underfitting.

### Example: Improved Image-Only Model Architecture

```python
pre_trained_model = EfficientNetB0(
    input_shape=image_shape, 
    include_top=False, 
    weights="imagenet", 
    name="mobilenet"
)
pre_trained_model.trainable = True
for layer in pre_trained_model.layers[:-20]:
    layer.trainable = False

data_augmentation = Sequential([
    RandomRotation(factor=0.15),
    RandomBrightness(factor=0.2),
    RandomZoom(height_factor=0.15, width_factor=0.15),
    RandomContrast(0.2),
    RandomFlip("horizontal_and_vertical"),
    GaussianNoise(stddev=0.02)
])

image_input = Input(shape=image_shape)
x = data_augmentation(image_input)
x = pre_trained_model(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(48, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
z = Dense(5, activation='softmax')(x)

model = Model(inputs=image_input, outputs=z)
```

For more details, see the [Configuration](#configuration) and [Usage Examples](#usage-examples) sections above.

## 🏗️ Tech Stack

- **Deep Learning**: TensorFlow 2.19+, Keras
- **Hyperparameter Tuning**: Keras Tuner
- **Data Processing**: NumPy, Pandas, scikit-learn
- **Image Processing**: OpenCV, Pillow
- **Visualization**: Matplotlib
- **Configuration**: Pydantic (mode-specific config classes)
- **CLI**: Typer, Click
- **Testing**: pytest
- **DevTools**: Pydantic, Black, Pylint, Flake8

## 📁 Project Structure

```
ImageProcessingClone/
├── src/
│   └── cleave_app/
│       ├── __init__.py
│       ├── main.py                 # CLI entry point
│       ├── config_schema.py        # Configuration validation (mode-specific)
│       ├── data_processing.py      # Data loading and preprocessing
│       ├── model_pipeline.py       # Model building and training
│       ├── hyperparameter_tuning.py # Hyperparameter optimization
│       ├── prediction_testing.py   # Model evaluation
│       └── grad_cam.py           # Model interpretability
├── tests/                         # Unit tests
├── notebooks/                     # Jupyter notebooks
├── config_files/                  # Example configurations
├── requirements.txt               # Dependencies
├── setup.py                      # Package configuration
└── README.md                     # This file
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new_feature`
3. Install in development mode: `pip install -e ".[dev]"`
4. Make your changes and add tests
5. Run tests: `pytest`
6. Commit your changes: `git commit -m 'Add new feature'`
7. Push to the branch: `git push origin feature/new_feature`
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- THORLABS for the Fiber Cleave Analyzer (FCA)
- TensorFlow and Keras 
- The open-source community for various tools and libraries

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/c-lombardi23/ImageProcessing/issues) page
2. Create a new issue with detailed information
3. Contact: clombardi23245@gmail.com

---

**Note**: This project is designed for research and development purposes. Please ensure you have proper data handling and validation procedures in place for production use.