import traceback
from typing import Callable, List

import pandas as pd


def _setup_callbacks(config, trainable_model) -> List[Callable]:
    """Setup training callbacks based on configuration.

    Args:
        config: Configuration object containing callback parameters
        trainable_model: Model instance that has callback creation methods

    Returns:
        List[Callable]: List of configured callbacks
    """
    callbacks = []

    # Setup checkpoint callback
    if (
        config.checkpoints == "y"
        and config.checkpoint_filepath
        and config.monitor
        and config.method
    ):
        checkpoint = trainable_model.create_checkpoints(
            config.checkpoint_filepath, config.monitor, config.method
        )
        callbacks.append(checkpoint)

    # Setup early stopping callback
    if (
        config.early_stopping == "y"
        and config.patience
        and config.monitor
        and config.method
    ):
        es = trainable_model.create_early_stopping(
            config.patience, config.method, config.monitor
        )
        callbacks.append(es)

    # Setup learning rate reduction callback
    if config.reduce_lr is not None:
        if config.reduce_lr_patience is not None:
            reduce_lr = trainable_model.reduce_on_plateau(
                patience=config.reduce_lr_patience,
                factor=config.reduce_lr,
            )
        else:
            reduce_lr = trainable_model.reduce_on_plateau(
                factor=config.reduce_lr
            )
        callbacks.append(reduce_lr)

    return callbacks


def _run_search_helper(
    config, tuner, train_ds, test_ds, best_params_path=None
) -> None:
    """Helper function for running hyperparameter search.

    Args:
        config: Configuration object
        tuner: Keras tuner instance
        train_ds: Training dataset
        test_ds: Test dataset
    """
    try:
        tuner.run_search(train_ds, test_ds)
        best_hp = tuner.get_best_hyperparameters().values
        print("Best hyperparameters:", best_hp)
        if best_params_path is not None:
            pd.DataFrame([best_hp]).to_csv(f"{best_params_path}")
        else:
            print("Best hyperparameters not saved")

        pathname = config.save_model_file
        if pathname is None:
            print("Model not saved - no path specified")
        else:
            if not pathname.endswith(".keras"):
                pathname = pathname + ".keras"
            tuner.save_best_model(pathname)
            print(f"Model saved to: {pathname}")

    except Exception as e:
        print(f"Error during hyperparameter search: {e}")
        traceback.print_exc()
        raise
