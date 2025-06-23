# Fiber Cleave Processing

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
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Description

This project implements a comprehensive machine learning pipeline for analyzing fiber cleave quality using images from the THORLABS Fiber Cleave Analyzer (FCA). The system consists of two main components:

1. **CNN Classification Model**: Classifies cleave images as good or bad based on visual features
2. **MLP Regression Model**: Predicts optimal tension parameters for producing good cleaves

The models use transfer learning with MobileNetV2 as the backbone and are optimized using Keras Tuner for hyperparameter optimization.

## ✨ Key Features

- **Transfer Learning**: Uses pre-trained MobileNetV2 for robust feature extraction
- **Multi-Modal Input**: Combines image features with numerical parameters
- **Hyperparameter Optimization**: Automated tuning using Keras Tuner
- **Flexible Architecture**: Supports both classification and regression tasks
- **K-Fold Cross Validation**: Robust model evaluation
- **GradCAM Visualization**: Model interpretability through attention maps
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
   - Create a CSV file with metadata (tension, angle, etc.)

2. **Create a configuration file** (see [Configuration](#configuration) section)

3. **Train a classification model**:
   ```bash
   cleave-app --file_path config.json
   ```

## ⚙️ Configuration

The application uses a JSON configuration file to specify all parameters. Here's an example:

```json
{
  "csv_path": "data/cleave_metadata.csv",
  "img_folder": "data/images/",
  "feature_scaler_path": "models/feature_scaler.pkl",
  "label_scaler_path": "models/label_scaler.pkl",
  "image_shape": [224, 224, 3],
  "feature_shape": [6],
  "test_size": 0.2,
  "buffer_size": 32,
  "batch_size": 8,
  "mode": "train_cnn",
  "learning_rate": 0.001,
  "model_path": "models/cleave_classifier.keras",
  "max_epochs": 50,
  "early_stopping": "y",
  "patience": 5
}
```

### Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `csv_path` | string | Path to CSV file with cleave metadata | Required |
| `img_folder` | string | Directory containing cleave images | Required |
| `feature_scaler_path` | string | Path to feature scaler (optional) | None |
| `label_scaler_path` | string | Path to label scaler (optional) | None |
| `image_shape` | list | Input image dimensions `[height, width, channels]` | `[224, 224, 3]` |
| `feature_shape` | list | Numerical feature dimensions | `[6]` for CNN, `[5]` for MLP |
| `test_size` | float | Fraction of data for testing | `0.2` |
| `buffer_size` | int | Buffer size for dataset shuffling | `32` |
| `batch_size` | int | Training batch size | `8` |
| `mode` | string | Operation mode (see modes below) | Required |
| `learning_rate` | float | Learning rate for optimization | `0.001` |
| `model_path` | string | Path to save/load model | Required |
| `max_epochs` | int | Maximum training epochs | `20` |
| `early_stopping` | string | Enable early stopping (`"y"`/`"n"`) | `"n"` |
| `patience` | int | Early stopping patience | `3` |

### Available Modes

| Mode | Description |
|------|-------------|
| `train_cnn` | Train CNN classification model |
| `train_mlp` | Train MLP regression model |
| `train_image_only` | Train image-only classification model |
| `cnn_hyperparameter` | Run hyperparameter search for CNN |
| `mlp_hyperparameter` | Run hyperparameter search for MLP |
| `image_hyperparameter` | Run hyperparameter search for image-only model |
| `test_cnn` | Test CNN model performance |
| `test_mlp` | Test MLP model performance |
| `train_kfold_cnn` | Train CNN with k-fold cross validation |
| `train_kfold_mlp` | Train MLP with k-fold cross validation |
| `grad_cam` | Generate GradCAM visualizations |

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

## 🏗️ Tech Stack

- **Deep Learning**: TensorFlow 2.19+, Keras
- **Hyperparameter Tuning**: Keras Tuner
- **Data Processing**: NumPy, Pandas, scikit-learn
- **Image Processing**: OpenCV, Pillow
- **Visualization**: Matplotlib
- **Configuration**: Pydantic
- **CLI**: Typer, Click
- **Testing**: pytest

## 📁 Project Structure

```
ImageProcessingClone/
├── src/
│   └── cleave_app/
│       ├── __init__.py
│       ├── main.py                 # CLI entry point
│       ├── config_schema.py        # Configuration validation
│       ├── data_processing.py      # Data loading and preprocessing
│       ├── model_pipeline.py       # Model building and training
│       ├── hyperparameter_tuning.py # Hyperparameter optimization
│       ├── prediction_testing.py   # Model evaluation
│       └── grad_cam.py            # Model interpretability
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
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install in development mode: `pip install -e ".[dev]"`
4. Make your changes and add tests
5. Run tests: `pytest`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- THORLABS for the Fiber Cleave Analyzer (FCA)
- TensorFlow and Keras teams for the excellent deep learning framework
- The open-source community for various tools and libraries

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/c-lombardi23/ImageProcessing/issues) page
2. Create a new issue with detailed information
3. Contact: clombardi23245@gmail.com

---

**Note**: This project is designed for research and development purposes. Please ensure you have proper data handling and validation procedures in place for production use.