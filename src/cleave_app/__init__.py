"""
Fiber Cleave Processing Package

A machine learning package for fiber cleave quality classification and tension prediction
using CNN and MLP models.
"""

__version__ = "0.1.0"
__author__ = "Chris Lombardi"
__email__ = "clombardi23245@gmail.com"

from .config_schema import Config
from .data_processing import DataCollector, MLPDataCollector
from .model_pipeline import CustomModel, BuildMLPModel
from .hyperparameter_tuning import (
    HyperParameterTuning, 
    ImageHyperparameterTuning, 
    MLPHyperparameterTuning,
    BuildHyperModel,
    ImageOnlyHyperModel,
    BuildMLPHyperModel
)
from .prediction_testing import TestPredictions, TensionPredictor
from .grad_cam import GradCAM, gradcam_driver, compute_saliency_map

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
    "compute_saliency_map"
]
