"""Prediction model pipeline for testing CNN model or MLP model.

This module provides classes for gathering data and then testing on
either the cnn model or the regression model.
"""

import os
from typing import List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

from .data_processing import BadCleaveTensionClassifier, DataCollector

# ====================================================
PRED_FEATURES = [
    "CleaveAngle",
    "CleaveTension",
    "ScribeDiameter",
    "Misting",
    "Hackle",
    # "Tearing",
]


class TestPredictions(DataCollector):
    """This class is used to test model performance on unseen data using
    metrics such as accuracy, precision, recall, and confusion matrix.

    Supports both image+feature and image-only CNNs.
    """

    def __init__(
        self,
        model_path: str,
        csv_path: str,
        scalar_path: str,
        img_folder: str,
        angle_threshold: float,
        diameter_threshold: float,
        encoder_path: str = None,
        image_only: bool = False,
        backbone: str = "mobilenet",
        classification_type: str = "binary",
        threshold: Optional[float] = 0.5,
    ):
        """Initialize TestPredictions.

        Args:
            model_path (str): Path to trained model.
            csv_path (str): Path to CSV file with metadata.
            scalar_path (str): Path to feature scaler (ignored if image_only=True).
            img_folder (str): Path to image folder.
            image_only (bool): If True, test only with images (no features).
        """
        DataCollector.__init__(
            self,
            csv_path,
            img_folder=img_folder,
            backbone=backbone,
            encoder_path=encoder_path,
            classification_type=classification_type,
            angle_threshold=angle_threshold,
            diameter_threshold=diameter_threshold,
        )
        if self.classification_type == "multiclass":
            self.class_names = self.encoder.categories_[0].tolist()
        elif self.classification_type == "binary":
            self.class_names = [0, 1]
        self.scalar_path = scalar_path
        self.model = tf.keras.models.load_model(model_path)
        self.model_path = model_path
        self.image_only = image_only
        self.ohe = None
        if not self.image_only and self.scalar_path:
            self.feature_scaler = joblib.load(self.scalar_path)
        self.threshold = threshold

    def _clean_data(self) -> "pd.DataFrame | None":
        """Read CSV file into DataFrame and add column for cleave quality and
        one-hot encoded labels.

        Returns:
            pd.DataFrame | None: DataFrame with cleave quality and one-hot labels, or None if file not found.
        """
        try:
            df = self._set_label(self.angle_threshold, self.diameter_threshold)
        except FileNotFoundError:
            print("CSV file not found!")
            return None
        # Clean image path
        df["ImagePath"] = df["ImagePath"].str.replace(
            f"{self.img_folder}\\", "", regex=False
        )
        # One-hot encode CleaveCategory
        if self.classification_type == "multiclass":
            self.ohe = joblib.load(self.encoder_path)
            onehot_labels = self.ohe.transform(df[["CleaveCategory"]])
            class_names = self.ohe.categories_[0]
            for idx, class_name in enumerate(class_names):
                df[f"Label_{class_name}"] = onehot_labels[:, idx]
            self.class_names = class_names
        return df

    def test_prediction(
        self,
        image_path: str,
        feature_vector: "np.ndarray | None" = None,
    ) -> "np.ndarray":
        """Generate prediction for a single image (and features if not
        image_only).

        Args:
            image_path (str): Path to image to predict.
            feature_vector (np.ndarray | None): Numerical features (ignored if image_only).

        Returns:
            np.ndarray: Model prediction.
        """
        image = self.load_process_images(image_path)
        image = np.expand_dims(image, axis=0)
        if self.image_only:
            prediction = self.model.predict(image)
        else:
            feature_vector = np.expand_dims(feature_vector, axis=0)
            feature_vector = np.zeros_like(feature_vector)
            prediction = self.model.predict([image, feature_vector])
        return prediction

    def gather_predictions(
        self,
    ) -> "tuple[np.ndarray, list, list] | tuple[None, None, None]":
        """Gather multiple predictions from test data.

        Returns:
            tuple: (true_labels, pred_labels, predictions) or (None, None, None) if no data.
        """
        if self.df is None:
            return None, None, None
        pred_image_paths = self.df["ImagePath"].values
        if self.image_only:
            pred_features = None
        else:
            pred_features = self.df[PRED_FEATURES].values
            # if self.scaler is not None:
            # pred_features = self.scaler.transform(pred_features)
        predictions = []
        if self.image_only:
            for img_path in pred_image_paths:
                prediction = self.test_prediction(img_path)
                predictions.append(prediction)
        else:
            if pred_features is not None:
                for img_path, feature_vector in zip(
                    pred_image_paths, pred_features
                ):
                    prediction = self.test_prediction(img_path, feature_vector)
                    predictions.append(prediction)
            else:
                print("No features available for prediction.")
                return None, None, None

        # Set prediction labels based on max
        pred_labels = [np.argmax(pred[0]) for pred in predictions]
        if self.classification_type == "binary":
            pred_labels = [
                (pred[0, 0] > self.threshold).astype(int)
                for pred in predictions
            ]
        elif self.classification_type == "multiclass":
            pred_labels = [np.argmax(pred[0]) for pred in predictions]

        true_labels = (
            self.df["CleaveCategory"]
            .map({label: idx for idx, label in enumerate(self.class_names)})
            .values
        )

        return true_labels, pred_labels, predictions

    def display_confusion_matrix(
        self,
        true_labels: "np.ndarray",
        pred_labels: "list[int]",
        model_path: str,
    ) -> None:
        """Display confusion matrix comparing true labels to predicted labels.

        Args:
            true_labels (np.ndarray): Array of true labels.
            pred_labels (list[int]): List of predicted labels.
        """
        if self.classification_type == "binary":
            labels = np.array([0, 1])
        elif self.classification_type == "multiclass":
            labels = list(range(len(self.class_names)))
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.class_names
        )
        disp.plot()

        model_dir = os.path.dirname(model_path)
        basename = os.path.basename(model_path)
        stem, _ = os.path.splitext(basename)
        save_confusion = os.path.join(
            model_dir, f"{stem}_confusion_report.png"
        )
        plt.savefig(save_confusion)
        plt.show()
        return save_confusion

    def display_classification_report(
        self,
        true_labels: "np.ndarray",
        pred_labels: "list[int]",
        classification_path: str = None,
    ) -> None:
        """Display classification report comparing true labels to predicted
        labels.

        Args:
            true_labels (np.ndarray): Array of true labels.
            pred_labels (list[int]): List of predicted labels.
            classification_path (str, optional): Optional path to save classification report.
        """
        if classification_path:
            df = pd.DataFrame(
                classification_report(
                    true_labels,
                    pred_labels,
                    target_names=self.class_names,
                    output_dict=True,
                )
            ).transpose()
            df.to_csv(classification_path, index=True)
        if self.classification_type == "binary":
            str_names = [str(c) for c in self.class_names]
            print(
                classification_report(
                    true_labels, pred_labels, target_names=str_names
                )
            )
        else:
            print(
                classification_report(
                    true_labels,
                    pred_labels,
                    target_names=self.class_names,
                )
            )

    def plot_roc(
        self,
        title: str,
        true_labels: "np.ndarray",
        pred_probabilites: "np.ndarray",
    ) -> None:
        """Plot ROC curve for predictions.

        Args:
            title (str): Title for the plot.
            true_labels (np.ndarray): Array of true labels.
            pred_probabilites (np.ndarray): Array of predicted probabilities.
        """
        pred_probabilites = np.array(pred_probabilites).flatten()
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probabilites)
        auc = roc_auc_score(true_labels, pred_probabilites)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc:.2f}%)")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        model_dir = os.path.dirname(self.model_path)
        basename = os.path.basename(self.model_path)
        stem, _ = os.path.splitext(basename)
        save_roc = os.path.join(model_dir, f"{stem}_roc_curve.png")
        plt.savefig(save_roc)
        plt.show()
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f"Optimal Threshold: {optimal_threshold:.2f}")
        return save_roc


class TestTensionPredictions(BadCleaveTensionClassifier):
    def __init__(
        self,
        cnn_model_path: str,
        tension_model_path: str,
        csv_path: str,
        scaler_path: str,
        img_folder: str,
        tension_threshold: int,
        image_only: bool,
    ) -> None:
        """Initialize the TestTensionPredictions pipeline.

        Args:
            cnn_model_path: Trained CNN classifier to identify good/bad cleave
            tension_model_path: Trained model to predict tension direction (raise/lower)
        """

        super().__init__(
            csv_path=csv_path,
            img_folder=img_folder,
            tension_threshold=tension_threshold,
            classification_type="binary",
            backbone="efficientnet",
            encoder_path=None,
        )
        self.tension_model = tf.keras.models.load_model(tension_model_path)
        self.image_only = image_only
        self.tester = TestPredictions(
            model_path=cnn_model_path,
            csv_path=csv_path,
            scalar_path=scaler_path,
            encoder_path=None,
            image_only=False,
            backbone="efficientnet",
            classification_type="binary",
            img_folder=img_folder,
        )

        if self.classification_type == "multiclass":
            self.class_names = self.encoder.categories_[0].tolist()
        elif self.classification_type == "binary":
            self.class_names = [0, 1]
        self.scalar_path = scaler_path
        self.ohe = None
        if not self.image_only and self.scalar_path:
            self.feature_scaler = joblib.load(self.scalar_path)

    def predict_tension(self, image_path: str, params=None):
        image = self.load_process_images(image_path)
        image = np.expand_dims(image, axis=0)
        tension_pred = self.tension_model.predict(
            [image, np.expand_dims(params, axis=0)]
        )
        direction = np.argmax(tension_pred)
        return (
            "Bad - raise tension" if direction == 1 else "Bad - lower tension"
        )

    def gather_predictions(self, pred_features=None):
        """Gather predictions for image-only or image+feature based
        classification/regression.

        Args:
            pred_image_paths (list): Paths to the images for prediction.
            pred_features (np.ndarray, optional): Feature vectors for each image.

        Returns:
            Tuple of (true_labels, pred_labels)
        """
        pred_image_paths = self.df["ImagePath"]

        pred_labels = []

        if not self.image_only:
            for img_path in pred_image_paths:
                feature_vector = np.zeros((6,), dtype=np.float32)
                classification = self.tester.test_prediction(
                    img_path, feature_vector
                )

                if classification < 0.5:
                    tension_pred = self.predict_tension(
                        img_path, feature_vector
                    )
                    pred_labels.append(tension_pred)
        elif self.image_only:
            if pred_features is not None:
                for img_path, feature_vector in zip(
                    pred_image_paths, pred_features
                ):
                    prediction = self.predict_tension(img_path, feature_vector)
                    pred_labels.append(prediction)
            else:
                for img_path in pred_image_paths:
                    feature_vector = np.zeros((6,), dtype=np.float32)
                    prediction = self.predict_tension(img_path, feature_vector)
                    pred_labels.append(prediction)

        if hasattr(self, "df") and "BadTensionsLabel" in self.df.columns:
            true_labels = self.df["BadTensionsLabel"].values
        else:
            print("True labels not available in dataframe.")
            return None, pred_labels
        print(len(pred_labels))
        return true_labels, pred_labels


class TensionPredictor:
    """Predicts tension values using a trained MLP model and preprocessed
    image/features."""

    def __init__(
        self,
        model_path: str,
        image_folder: str,
        csv_path: str,
        angle_threshold: float,
        diameter_threshold: float,
        tension_scaler_path: Optional[str] = None,
    ):
        """Initialize TensionPredictor.

        Args:
            model (tf.keras.Model): Trained MLP model for tension prediction.
            image_folder (str): Path to image folder.
            image_path (str): Path to image for prediction.
            angle_threshold: Maximum angle for good cleave.
            diameter_threshold: Maximum diameter for good cleave.
            tension_scaler_path (str): Path to saved tension scaler.
        """
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)
        self.image_folder = image_folder
        if tension_scaler_path:
            self.tension_scaler = joblib.load(tension_scaler_path)
        self.csv_path = csv_path
        self.angle_threshold = angle_threshold
        self.diameter_threshold = diameter_threshold

    def load_and_preprocess_image(
        self, file_path: str, img_folder: str
    ) -> "tf.Tensor":
        """Load and preprocess image from file path.

        Args:
            file_path (str): Path to image file.
            img_folder (str): Path to image folder.

        Returns:
            tf.Tensor: Preprocessed image tensor.
        """
        # Construct full path
        full_path = os.path.join(img_folder, file_path)
        img_raw = tf.io.read_file(full_path)
        img = tf.image.decode_png(img_raw, channels=1)
        img = tf.image.resize(img, [224, 224])
        img = tf.image.grayscale_to_rgb(img)
        # Normalize image
        return img

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
                and row["ScribeDiameter"] < diameter_threshold * row['Diameter']
                and not row["Hackle"]
                and not row["Misting"]
                else 0
            ),
            axis=1,
        )
        # Compute mean tension from good cleaves
        good_mean = df[df["CleaveCategory"] == 1]["CleaveTension"].mean()

        # Keep only bad cleaves
        bad_df = df[df["CleaveCategory"] == 0].copy()

        # Compute true delta (label) = good_mean - current
        bad_df["TrueDelta"] = good_mean - bad_df["CleaveTension"]

        return bad_df, good_mean

    def predict(self):
        """Run tension predictions on filtered cleave data."""
        if not self.model or not self.tension_scaler:
            raise RuntimeError(
                "Model and scaler must be loaded before prediction."
            )

        df, mean = self._extract_data(
            angle_threshold=self.angle_threshold,
            diameter_threshold=self.diameter_threshold,
        )
        image_paths = df["ImagePath"]
        tensions = df["CleaveTension"]
        true_delta = df["TrueDelta"]

        predictions = []
        predicted_deltas = []
        pred_ts = []

        for img_path in image_paths:
            image = self.load_and_preprocess_image(img_path, self.image_folder)
            image = tf.expand_dims(image, 0)

            features = np.zeros((4,))
            features = features.reshape(1, -1)

            pred_scaled = self.model.predict([image, features])[0]
            delta = self.tension_scaler.inverse_transform(
                pred_scaled.reshape(1, -1)
            )[0][0]
            predicted_deltas.append(delta)
            predictions.append(delta + tensions.iloc[len(predictions)])

        for true_t, delta_pred, current_t in zip(
            true_delta, predicted_deltas, tensions
        ):
            pred_t = current_t + delta_pred
            pred_ts.append(pred_t)
            print(
                f"Current: {current_t:.2f} | True delta: {true_t:.2f} | Pred delta: {delta_pred:.2f} | Pred T: {pred_t:.2f} | Target T: {mean:.2f}"
            )

        df = pd.DataFrame(
            {
                "Current Tension": np.array(tensions).round(2),
                "True Delta": np.array(true_delta).round(2),
                "Predicted Tension": np.array(predictions).round(2),
                "Predicted Delta": np.array(predicted_deltas).round(2)
            }
        )
        basepath = self.model_path.strip(".keras")
        df.to_csv(f"{basepath}_performance.csv", index=False)

        return tensions, true_delta, predicted_deltas, pred_ts, mean

    def plot_metric(
        self,
        title: str,
        X: "list[float]",
        y: "list[float]",
        x_label: str,
        y_label: str,
        x_legend: str,
        y_legend: str,
    ) -> None:
        """Plot a metric for model evaluation.

        Args:
            title (str): Title of plot.
            X (list[float]): List of x values.
            y (list[float]): List of y values.
            x_label (str): Label for x axis.
            y_label (str): Label for y axis.
            x_legend (str): Legend for x axis.
            y_legend (str): Legend for y axis.
        """
        plt.title(title)
        plt.plot(X, label=x_legend)
        plt.plot(y, label=y_legend)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="lower right")
        plt.show()
