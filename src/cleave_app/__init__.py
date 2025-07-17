"""
Fiber Cleave Processing Package

A machine learning package for fiber cleave quality classification and tension prediction
using CNN and MLP models.
"""

__version__ = "0.1.0"
__author__ = "Chris Lombardi"
__email__ = "clombardi23245@gmail.com"

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from .data_processing import DataCollector, MLPDataCollector
from .grad_cam import GradCAM, compute_saliency_map, gradcam_driver
from .hyperparameter_tuning import (
    BuildHyperModel,
    BuildMLPHyperModel,
    HyperParameterTuning,
    ImageHyperparameterTuning,
    ImageOnlyHyperModel,
    MLPHyperparameterTuning,
)
from .mlp_model import BuildMLPModel
from .model_pipeline import CustomModel
from .prediction_testing import TensionPredictor, TestPredictions
from .xgboost_pipeline import XGBoostModel, XGBoostPredictor

__all__ = [
    "Config",
    "DataCollector",
    "MLPDataCollector",
    "CustomModel",
    "BuildMLPModel",
    "HyperParameterTuning",
    "ImageHyperparameterTuning",
    "MLPHyperparameterTuning",
    "BuildHyperModel",
    "ImageOnlyHyperModel",
    "BuildMLPHyperModel",
    "TestPredictions",
    "TensionPredictor",
    "GradCAM",
    "gradcam_driver",
    "compute_saliency_map",
    "XGBoostModel",
    "XGBoostPredictor",
]
