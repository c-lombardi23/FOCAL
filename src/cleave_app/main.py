"""Main CLI module for the Fiber Cleave Processing application.

This module provides the command-line interface for training and testing
CNN and MLP models for fiber cleave quality classification and tension
prediction.
"""

import argparse
import os
import sys
import traceback
import warnings
from typing import Optional

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Import application modules
from .config_schema import load_config

from .commands.train_commands import (
    TrainCNN,
    TrainMLP,
    TrainXGBoost,
    TrainImageOnly,
    KFoldCNN,
    KFoldMLP,
    TrainCustomModel,
)
from .commands.test_commands import (
    TestCNN,
    TestMLP,
    TestImageOnly,
    TestXGBoost,
)
from .commands.hyperparameter_commands import (
    CNNHyperparameterSearch,
    MLPHyperparameterSearch,
    ImageHyperparameterSearch,
)
from .commands.grad_cam_commands import GradCamDisplay

try:
    import tensorflow as tf
except ImportError:
    print("Warning: TensorFlow not found. Please install tensorflow>=2.19.0")
    traceback.print_exc()
    tf = None


def main(args: Optional[list] = None) -> int:
    """Main entry point for the CLI application.

    Args:
        args: Command line arguments (optional)

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Fiber Cleave Processing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cleave-app --file_path config.json
  cleave-app --file_path /path/to/config.json
        """,
    )

    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to JSON configuration file",
    )

    parsed_args = parser.parse_args(args)

    try:
        config = load_config(parsed_args.file_path)

        command_map = {
            "train_cnn": TrainCNN,
            "train_mlp": TrainMLP,
            "train_xgboost": TrainXGBoost,
            "train_image_only": TrainImageOnly,
            "train_kfold_cnn": KFoldCNN,
            "train_kfold_mlp": KFoldMLP,
            "custom_model": TrainCustomModel,
            "test_cnn": TestCNN,
            "test_mlp": TestMLP,
            "test_image_only": TestImageOnly,
            "test_xgboost": TestXGBoost,
            "cnn_hyperparameter": CNNHyperparameterSearch,
            "mlp_hyperparameter": MLPHyperparameterSearch,
            "image_hyperparameter": ImageHyperparameterSearch,
            "grad_cam": GradCamDisplay,
        }
        command_class = command_map.get(config.mode)

        if command_class is None:
            raise ValueError(
                f"Unknown mode: '{config.mode}'. "
                f"Please check your configuration file. "
                f"Available modes are: {list(command_map.keys())}"
            )

        command_instance = command_class()
        command_instance.execute(config)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
