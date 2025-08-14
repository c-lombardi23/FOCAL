"""Main module for all logic related to XGBoost.

Includes classes for training and predicting.
"""

import os
from typing import Any, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model


class XGBoostModel:
    """This class provides basic logic for training the XGBoost regressor."""

    def __init__(
        self, csv_path: str, cnn_model_path: str, train_ds: Any, test_ds: Any
    ) -> None:
        """Initialize the model.

        Args:
            csv_path (str): _Path to the csv dataset.
            cnn_model_path (str): Path to the trained CNN classifier.
            train_ds (tf.data.Dataset): Train split of dataset.
            test_ds (tf.data.Dataset): Test split of dataset.
        """
        self.csv_path = csv_path
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.xgb_reg = None
        self.xgb_path = None

        try:
            self.cnn_model = tf.keras.models.load_model(cnn_model_path)
            self.feature_extractor = Model(
                inputs=self.cnn_model.input[0],
                outputs=self.cnn_model.get_layer("global_avg").output,
            )
        except ValueError as e:
            print(f"Error loading model or extracting layer: {e}")
            self.feature_extractor = None

    def get_model(self):
        return self.xgb_reg

    def _extract_features_and_labels(
        self, ds: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract image features and delta tension values from the dataset
        using feature extractor.

        Args:
            ds: Dataset to be extracted.

        Returns:
            Tuple of features and delta tension values
        """
        features, delta_tensions = [], []
        for img_batch, tension_batch in ds:
            feats = self.feature_extractor(img_batch).numpy()
            features.append(feats)
            delta_tensions.append(tension_batch.numpy().reshape(-1))
        return np.vstack(features), np.concatenate(delta_tensions)

    def train(
        self,
        error_type="reg:squarederror",
        n_estimators: Optional[int] = 200,
        learning_rate: Optional[float] = 0.05,
        max_depth: Optional[int] = 4,
        random_state: Optional[int] = 42,
        gamma: Optional[float] = 0.0,
        subsample: Optional[float] = 1.0,
        reg_lambda: Optional[float] = 1.0,
    ):
        """Training logic for the xgboost regression model.

        Args:
            n_estimators: Maximum number of trees to use during training
            learning_rate: Learning rate to update weights during training
            max_depth: Maximum tree depth during training
            random_state: Controls random state of model to ensure consitency across models
            gamma: Minimum loss reduction.
            subsample: Fraction of observations used for each tree.
            reg_lambda: L2 regularization of leaf nodes.

        Returns:
            Trained xgboost model
        """

        X_train, y_train = self._extract_features_and_labels(self.train_ds)
        X_test, y_test = self._extract_features_and_labels(self.test_ds)

        self.xgb_reg = xgb.XGBRegressor(
            objective=error_type,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            gamma=gamma,
            subsample=subsample,
            reg_lambda=reg_lambda,
        )

        self.xgb_reg.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=True,
        )

        y_pred = self.xgb_reg.predict(X_test)
        print("Val MAE:", mean_absolute_error(y_test, y_pred))

        return self.xgb_reg.evals_result()

    def save(self, save_path: str) -> None:
        """Saves xgboost model.

        Args:
            save_path: Path to save model.

        Raises:
            ValueError: If trained model is None.
        """
        if self.xgb_reg is not None:
            joblib.dump(self.xgb_reg, save_path)
            self.xgb_path = save_path
        else:
            raise ValueError("Model not trained yet.")

    def load(self, model_path: str):
        """Load model from path."""
        self.xgb_reg = joblib.load(model_path)

    def plot_metrics(
        self,
        title: str,
        metric1,
        metric2,
        metric1_label: str,
        metric2_label: str,
        x_label: str,
        y_label: str,
    ) -> None:
        """Basic plotting function for viewing metrics.

        Args:
            title: Title of metric plot
            metric1: First metric to plot
            metric2: Second metric to plot
            metric1_label: Metric 1 identifying label
            metric2_labe: Metric 2 identifying label
            x_label: X-axis label
            y-Label: Y_axis label
        """
        plt.title(title)
        plt.plot(metric1, label=metric1_label)
        plt.plot(metric2, label=metric2_label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper right")
        model_dir = self.xgb_path.parent.parent
        basename = self.xgb_path.name
        save_path = model_dir / "metrics" / basename
        save_plot = f"{save_path}_{title}.png"
        plt.savefig(save_plot)
        plt.show()


class XGBoostPredictor:
    """This class implements basic logic for predicting and testing the change
    in tensions."""

    def __init__(
        self,
        csv_path: str,
        cnn_model_path: str,
        angle_threshold: float,
        diameter_threshold: float,
        xgb_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
    ) -> None:
        """Initialize predictor class.

        Args:
            csv_path (str): Path to csv file for dataset.
            cnn_model_path (str): Path to trained CNN classifier.
            angle_threshold (float): Maximum angle for classifying as good cleave.
            diameter_threshold (float): Maximum diameter for classifying as good cleave.
            xgb_path (Optional[str], optional): Path to trained XGBoost regressor.
            scaler_path (Optional[str], optional): Path to tensions scaler.
        """
        self.csv_path = csv_path
        self.xgb_path = xgb_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.angle_threshold = angle_threshold
        self.diameter_threshold = diameter_threshold

        # Load CNN
        try:
            self.cnn_model = tf.keras.models.load_model(cnn_model_path)
            self.feature_extractor = Model(
                inputs=self.cnn_model.input[0],
                outputs=self.cnn_model.get_layer("global_avg").output,
            )
        except (OSError, ValueError) as e:
            print(f"[CNN Load Error]: {e}")
            self.feature_extractor = None

    def _extract_cnn_features(self, image_path: str) -> np.ndarray:
        """Extract CNN features from a grayscale image.

        Args:
            image_path: Path to image to extract features

        Returns:
            Extracted features from image.
        """
        img_raw = tf.io.read_file(image_path)
        img = tf.image.decode_png(img_raw, channels=1)
        img = tf.image.resize(img, [224, 224])
        img = tf.image.grayscale_to_rgb(img)
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img, axis=0)
        # remove single dimensional entries
        return self.feature_extractor(img).numpy().squeeze()

    def _extract_data(self, angle_threshold: float, diameter_threshold: float):
        """Load and filter dataset for prediction (only bad cleaves).

        Args:
            angle_threshold: Maximum angle to be considered good cleave
            diameter_threshold: Maximum diameter of scribe mark to be considered good cleave

        Returns:
            Df of bad cleaves and the mean of the good cleaves

        """
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV: {e}") from e

        df["CleaveCategory"] = df.apply(
            lambda row: (
                1
                if row["CleaveAngle"] <= angle_threshold
                and row["ScribeDiameter"]
                < diameter_threshold * row["Diameter"]
                and not row["Hackle"]
                and not row["Misting"]
                else 0
            ),
            axis=1,
        )

        # Compute mean tension from good cleaves
        # good_mean = df[df["CleaveCategory"] == 1]["CleaveTension"].mean()
        good_cleaves_df = df[df["CleaveCategory"] == 1]
        mean_tension_per_type = (
            good_cleaves_df.groupby("FiberType")["CleaveTension"]
            .mean()
            .to_dict()
        )
        # Keep only bad cleaves
        bad_df = df[df["CleaveCategory"] == 0].copy()
        bad_df["FiberTypeMeanTension"] = bad_df["FiberType"].map(
            mean_tension_per_type
        )

        # Compute true delta (label) = good_mean - current
        bad_df["TrueDelta"] = (
            bad_df["FiberTypeMeanTension"] - bad_df["CleaveTension"]
        )

        return bad_df, mean_tension_per_type

    def load(self):
        """Load trained model and scaler."""
        if not self.xgb_path or not self.scaler_path:
            raise ValueError("Paths for model and scaler must be provided.")

        try:
            self.model = joblib.load(self.xgb_path)
            self.scaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or scaler: {e}") from e

    def predict(self):
        """Run tension predictions on filtered cleave data."""
        if not self.model or not self.scaler:
            raise RuntimeError(
                "Model and scaler must be loaded before prediction."
            )

        df, mean_tensions = self._extract_data(
            angle_threshold=self.angle_threshold,
            diameter_threshold=self.diameter_threshold,
        )
        image_paths = df["ImagePath"]
        tensions = df["CleaveTension"]
        true_delta = df["TrueDelta"]
        fiber_type = df["FiberType"]

        predictions = []
        predicted_deltas = []
        pred_ts = []

        for img_path in image_paths:
            features = self._extract_cnn_features(img_path)
            pred_scaled = self.model.predict(features.reshape(1, -1))[0]
            delta = self.scaler.inverse_transform([[pred_scaled]])[0][0]
            predicted_deltas.append(delta)
            predictions.append(delta + tensions.iloc[len(predictions)])

        for true_t, delta_pred, current_t, fiber in zip(
            true_delta, predicted_deltas, tensions, fiber_type
        ):
            pred_t = current_t + delta_pred
            pred_ts.append(pred_t)

            current_mean = mean_tensions[fiber]
            print(
                f"Current: {current_t:.2f} | True delta: {true_t:.2f} | Pred delta: {delta_pred:.2f} | Pred T: {pred_t:.2f} | Target T: {current_mean:.2f}"
            )

        df = pd.DataFrame(
            {
                "Current Tension": np.array(tensions).round(2),
                "True Delta": np.array(true_delta).round(2),
                "Predicted Tension": np.array(predictions).round(2),
                "Predicted Delta": np.array(predicted_deltas).round(2),
            }
        )

        return (
            tensions,
            predicted_deltas,
            predictions,
            true_delta,
            mean_tensions,
        )
