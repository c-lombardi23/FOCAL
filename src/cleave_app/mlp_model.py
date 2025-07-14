"""Main module for defining MLP training logic."""

from .model_pipeline import *


class BuildMLPModel(CustomModel):
    """MLP model for tension prediction using features from pre-trained CNN.

    This class builds regression models that use features extracted from
    a CNN to predict optimal tension values for fiber cleaving.
    """

    def __init__(self, cnn_model_path: str, train_ds, test_ds, num_classes):
        super().__init__(train_ds, test_ds, num_classes=num_classes)
        # load your frozen CNN
        self.cnn_model = tf.keras.models.load_model(cnn_model_path)
        self.feature_output = self.cnn_model.get_layer("global_avg").output
        self.image_input = self.cnn_model.input[0]

    def _build_pretrained_model(
        self,
        param_shape: Tuple[int, ...],
        dense1: int,
        dense2: int,
        dropout1: float,
        dropout2: float,
        dropout3: float,
    ) -> tf.keras.Model:
        """Build MLP model for tension prediction.

        Args:
            param_shape: Dimensions of numerical parameters

        Returns:
            tf.keras.Model: Regression model for tension prediction
        """
        x = Dropout(dropout1, name="dropout_1")(self.feature_output)
        feature_input = Input(shape=param_shape, name="feature_input")

        y = Dense(dense1, name="dense_1", activation="relu")(feature_input)
        y = Dropout(dropout2, name="dropout_2")(y)

        combined = Concatenate()([x, y])
        z = Dense(dense2, name="dense_2", activation="relu")(combined)
        z = Dropout(dropout3, name="dropout3")(z)

        output = Dense(1, name="tension_output")(z)
        regression_model = Model(
            inputs=[self.image_input, feature_input], outputs=output
        )
        regression_model.summary()
        return regression_model

    def compile_model(
        self,
        dense1: int,
        dense2: int,
        dropout1: float,
        dropout2: float,
        dropout3: float,
        param_shape: Tuple[int, ...],
        learning_rate: float = 0.001,
        metrics: Optional[List[str]] = None,
    ) -> "tf.keras.Model":
        """Compile MLP model for regression.

        Args:
            param_shape: Dimensions of numerical parameters
            learning_rate: Learning rate for optimization
            metrics: List of metrics to monitor

        Returns:
            tf.keras.Model: Compiled regression model
        """
        if metrics is None:
            metrics = ["mae"]

        model = self._build_pretrained_model(
            param_shape=param_shape,
            dense1=dense1,
            dense2=dense2,
            dropout1=dropout1,
            dropout2=dropout2,
            dropout3=dropout3,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=metrics)
        return model

    def create_early_stopping(
        self,
        patience: int = 3,
        mode: str = "min",
        monitor: str = "val_mae",
    ) -> "EarlyStopping":
        """Create early stopping callback for regression model.

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
        """Create model checkpoints for MLP model.

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
        num_classes: int,
        dense1: int,
        dense2: int,
        dropout1: float,
        dropout2: float,
        dropout3: float,
        checkpoints: Optional[ModelCheckpoint] = None,
        epochs: int = 5,
        initial_epoch: int = 0,
        early_stopping: Optional[EarlyStopping] = None,
        history_file: Optional[str] = None,
        save_model_file: Optional[str] = None,
    ) -> Tuple[List[tf.keras.Model], List[tf.keras.callbacks.History]]:
        """Train MLP model using k-fold cross validation.

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

            custom_model = BuildMLPModel(cnn_model_path, train_ds, test_ds, num_classes)
            model = custom_model.compile_model(
                param_shape=param_shape,
                learning_rate=learning_rate,
                dense1=dense1,
                dense2=dense2,
                dropout1=dropout1,
                dropout2=dropout2,
                dropout3=dropout3,
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

            if save_model_file:
                os.makedirs(os.path.dirname(save_model_file), exist_ok=True)
                save_model_file = save_model_file.strip(".keras")
                model.save(f"{save_model_file.strip}_fold{fold+1}.keras")
                print(f"Fold {fold+1} model saved")
            else:
                print("Model not saved")

        return k_models, kfold_histories

    @staticmethod
    def get_averages_from_kfold(
        kfold_histories: List[tf.keras.callbacks.History], scaler: any
    ) -> None:
        """Calculate and display average metrics from k-fold cross validation
        for MLP.

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
