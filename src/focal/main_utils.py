import os
from pathlib import Path
from typing import Any

def resolve_paths(config: Any) -> Any:

    if not config.output_dir:
        return config
    
    model_dir = "models"
    scaler_dir = "scalers"
    metrics_dir = "metrics"

    scaler_keys = [
        "feature_scaler_path",
        "label_scaler_path",
        "tension_scaler_path",
        "encoder_path"
    ]

    output_dir = Path(config.output_dir)

    filepath_keys = [
        "csv_path",
        "img_folder",
        "tuner_directory"
    ]
    model_keys = [
        "model_path",
        "xgb_path",
        "checkpoints_filepath",
        "save_model_file",
        "cnn_path",
        "agent_path"
    ]
    metrics_keys = [
        "save_history_file",
        "classification_path",
        "best_params_path",
        "save_path"]

    for key in scaler_keys:
        if hasattr(config, key):

            relative_path = getattr(config, key)
            if relative_path:
                full_path = output_dir / scaler_dir / relative_path
                setattr(config, key, full_path)
    
    for key in filepath_keys:
        if hasattr(config, key):

            relative_path = getattr(config, key)
            if relative_path:
                full_path = output_dir / relative_path
                setattr(config, key, full_path)

    for key in model_keys:
        if hasattr(config, key):

            relative_path = getattr(config, key)
            if relative_path:
                full_path = output_dir / model_dir / relative_path
                setattr(config, key, full_path)

    for key in metrics_keys:
        if hasattr(config, key):

            relative_path = getattr(config, key)
            if relative_path:
                full_path = output_dir / metrics_dir / relative_path
                setattr(config, key, full_path)

    return config