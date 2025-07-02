from .model_pipeline import *
import xgboost as xgb
import joblib


class BuildMLPModel(CustomModel):
    """
    MLP model for tension prediction using features from pre-trained CNN.

    This class builds regression models that use features extracted from a CNN
    to predict optimal tension values for fiber cleaving.
    """

    def __init__(self, cnn_model_path: str, train_ds, test_ds):
        super().__init__(train_ds, test_ds)
        # load your frozen CNN
        self.cnn_model = tf.keras.models.load_model(cnn_model_path)

    def _build_pretrained_model(
        self, param_shape: Tuple[int, ...]
    ) -> tf.keras.Model:
        """
        Build MLP model for tension prediction.

        Args:
            param_shape: Dimensions of numerical parameters

        Returns:
            tf.keras.Model: Regression model for tension prediction
        """
        x = Dense(64, name="first_dense_layer", activation="relu")(
            self.feature_output
        )
        x = Dense(32, name="second_dense_layer", activation="relu")(x)
        feature_input = Input(shape=param_shape, name="feature_input")
        y = Dense(16, name="third_dense_layer", activation="relu")(
            feature_input
        )

        combined = Concatenate()([x, y])
        z = Dense(64, activation="relu")(combined)
        output = Dense(1, name="tension_output")(z)
        regression_model = Model(
            inputs=[self.image_input, feature_input], outputs=output
        )
        regression_model.summary()
        return regression_model

    def compile_model(
        self,
        param_shape: Tuple[int, ...],
        learning_rate: float = 0.001,
        metrics: Optional[List[str]] = None,
    ) -> "tf.keras.Model":
        """
        Compile MLP model for regression.

        Args:
            param_shape: Dimensions of numerical parameters
            learning_rate: Learning rate for optimization
            metrics: List of metrics to monitor

        Returns:
            tf.keras.Model: Compiled regression model
        """
        if metrics is None:
            metrics = ["mae"]

        model = self._build_pretrained_model(param_shape)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=metrics)
        return model

    def create_early_stopping(
        self,
        patience: int = 3,
        mode: str = "min",
        monitor: str = "val_mae",
    ) -> "EarlyStopping":
        """
        Create early stopping callback for regression model.

        Args:
            patience: Number of epochs to wait before stopping
            mode: Method to track monitor (min for regression)
            monitor: Metric to monitor during training

        Returns:
            tf.keras.callbacks.EarlyStopping: Early stopping callback
        """
        es_callback = EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            restore_best_weights=True,
            verbose=1,
        )
        return es_callback

    def create_checkpoints(
        self,
        checkpoint_filepath: str = "./checkpoints/mlp_model.keras",
        monitor: str = "val_mae",
        mode: str = "min",
        save_best_only: bool = True,
    ) -> ModelCheckpoint:
        """
        Create model checkpoints for MLP model.

        Args:
            checkpoint_filepath: Path to save model checkpoints
            monitor: Metric to monitor during training
            mode: Method to determine stopping point (min for regression)
            save_best_only: Whether to save only the best model

        Returns:
            tf.keras.callbacks.ModelCheckpoint: Checkpoint callback
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)

        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only,
            verbose=1,
        )
        return model_checkpoint_callback

    @staticmethod
    def train_kfold_mlp(
        datasets: List[Tuple],
        cnn_model_path: str,
        param_shape: Tuple[int, ...],
        learning_rate: float,
        checkpoints: Optional[ModelCheckpoint] = None,
        epochs: int = 5,
        initial_epoch: int = 0,
        early_stopping: Optional[EarlyStopping] = None,
        history_file: Optional[str] = None,
        model_file: Optional[str] = None,
    ) -> Tuple[List[tf.keras.Model], List[tf.keras.callbacks.History]]:
        """
        Train MLP model using k-fold cross validation.

        Args:
            datasets: List of (train_ds, test_ds) tuples for each fold
            cnn_model_path: Path to the pre-trained CNN model
            param_shape: Dimensions of numerical parameters
            learning_rate: Learning rate for optimization
            checkpoints: Checkpoint callback
            epochs: Number of training epochs
            initial_epoch: Starting epoch number
            early_stopping: Early stopping callback
            history_file: Base filename for saving training history
            model_file: Base filename for saving models

        Returns:
            Tuple of (list of trained models, list of training histories)
        """
        kfold_histories = []
        k_models = []
        train_datasets = [i[0] for i in datasets]
        test_datasets = [i[1] for i in datasets]

        callbacks = []
        if early_stopping:
            callbacks.append(early_stopping)
        if checkpoints:
            callbacks.append(checkpoints)

        for fold, (train_ds, test_ds) in enumerate(
            zip(train_datasets, test_datasets)
        ):
            print(f"\n=== Training MLP fold {fold + 1} ===")

            custom_model = BuildMLPModel(cnn_model_path, train_ds, test_ds)
            model = custom_model.compile_model(
                param_shape=param_shape,
                learning_rate=learning_rate,
                metrics=["mae", "mse"],
            )

            if callbacks:
                history = model.fit(
                    train_ds,
                    epochs=epochs,
                    initial_epoch=initial_epoch,
                    validation_data=test_ds,
                    callbacks=callbacks,
                )
            else:
                print("Training without callbacks")
                history = model.fit(
                    train_ds,
                    epochs=epochs,
                    initial_epoch=initial_epoch,
                    validation_data=test_ds,
                )

            kfold_histories.append(history)
            k_models.append(model)

            # Save fold-specific history and model
            if history_file:
                os.makedirs(os.path.dirname(history_file), exist_ok=True)
                df = pd.DataFrame(history.history)
                df.to_csv(f"{history_file}_fold{fold+1}.csv", index=False)
                print(f"Fold {fold+1} history saved")
            else:
                print("History not saved")

            if model_file:
                os.makedirs(os.path.dirname(model_file), exist_ok=True)
                model.save(f"{model_file}_fold{fold+1}.keras")
                print(f"Fold {fold+1} model saved")
            else:
                print("Model not saved")

        return k_models, kfold_histories

    @staticmethod
    def get_averages_from_kfold(
        kfold_histories: List[tf.keras.callbacks.History], scaler: Any
    ) -> None:
        """
        Calculate and display average metrics from k-fold cross validation for MLP.

        Args:
            kfold_histories: List of training histories from k-fold training
            scaler: Scaler used for tension values (for denormalization)
        """
        mae = []
        mse = []

        max_tension = scaler.data_max_[0]
        min_tension = scaler.data_min_[0]

        for history in kfold_histories:
            mae.append(min(history.history["mae"]))
            mse.append(min(history.history["mse"]))

        avg_mae = np.mean(mae)
        avg_mse = np.mean(mse)

        # Denormalize metrics
        mae_val = avg_mae * (max_tension - min_tension)
        mse_val = avg_mse * (max_tension - min_tension) ** 2
        rmse_val = np.sqrt(mse_val)

        print(f"Average MAE: {mae_val:.2f}")
        print(f"Average RMSE: {rmse_val:.2f}")
