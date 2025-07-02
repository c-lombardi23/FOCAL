"""
Prediction model pipeline for testing CNN model or MLP model.

This module provides classes for gathering data and then testing on either
the cnn model or the regression model.
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from .data_processing import DataCollector, BadCleaveTensionClassifier
from typing import List, Optional



class TestPredictions(DataCollector):
    """
    This class is used to test model performance on unseen data using metrics such as
    accuracy, precision, recall, and confusion matrix. Supports both image+feature and image-only CNNs.
    """

    def __init__(
        self,
        model_path: str,
        csv_path: str,
        scalar_path: str,
        img_folder: str,
        encoder_path: str = None,
        image_only: bool = False,
        backbone: str = "mobilenet",
        classification_type: str = "binary",
    ):
        """
        Initialize TestPredictions.

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
        )
        if self.classification_type == "multiclass":
            self.class_names = self.encoder.categories_[0].tolist()
        elif self.classification_type == "binary":
            self.class_names = [0, 1]
        self.scalar_path = scalar_path
        self.model = tf.keras.models.load_model(model_path)
        self.image_only = image_only
        self.ohe = None
        if not self.image_only and self.scalar_path:
            self.feature_scaler = joblib.load(self.scalar_path)

    def _clean_data(self) -> "pd.DataFrame | None":
        """
        Read CSV file into DataFrame and add column for cleave quality and one-hot encoded labels.

        Returns:
            pd.DataFrame | None: DataFrame with cleave quality and one-hot labels, or None if file not found.
        """
        try:
            df = self._set_label()
        except FileNotFoundError:
            print("CSV file not found!")
            return None
        # Clean image path
        df["ImagePath"] = df["ImagePath"].str.replace(
            self.img_folder, "", regex=False
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
        """
        Generate prediction for a single image (and features if not image_only).

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
        """
        Gather multiple predictions from test data.

        Returns:
            tuple: (true_labels, pred_labels, predictions) or (None, None, None) if no data.
        """
        if self.df is None:
            return None, None, None
        pred_image_paths = self.df["ImagePath"].values
        if self.image_only:
            pred_features = None
        else:
            pred_features = self.df[
                [
                    "CleaveAngle",
                    "CleaveTension",
                    "ScribeDiameter",
                    "Misting",
                    "Hackle",
                    "Tearing",
                ]
            ].values
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

        # Set prediction labels based on max of ohe
        pred_labels = [np.argmax(pred[0]) for pred in predictions]
        if self.classification_type == "binary":
            pred_labels = [
                (pred[0, 0] > 0.95).astype(int) for pred in predictions
            ]
        elif self.classification_type == "multiclass":
            pred_labels = [np.argmax(pred[0]) for pred in predictions]

        true_labels = (
            self.df["CleaveCategory"]
            .map({label: idx for idx, label in enumerate(self.class_names)})
            .values
        )

        return true_labels, pred_labels

    def display_confusion_matrix(
        self,
        true_labels: "np.ndarray",
        pred_labels: "list[int]",
        model_path: str,
    ) -> None:
        """
        Display confusion matrix comparing true labels to predicted labels.

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

    def display_classification_report(
        self,
        true_labels: "np.ndarray",
        pred_labels: "list[int]",
        classification_path: str = None,
    ) -> None:
        """
        Display classification report comparing true labels to predicted labels.

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
        """
        Plot ROC curve for predictions.

        Args:
            title (str): Title for the plot.
            true_labels (np.ndarray): Array of true labels.
            pred_probabilites (np.ndarray): Array of predicted probabilities.
        """
        pred_probabilites = np.array(pred_probabilites).flatten()
        fpr, tpr = roc_curve(true_labels, pred_probabilites)
        auc = roc_auc_score(true_labels, pred_probabilites)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc:.2f}%)")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()



class TestTensionPredictions(BadCleaveTensionClassifier):
    def __init__(self, 
                 cnn_model_path: str, 
                 tension_model_path: str,
                 csv_path: str,
                 scaler_path: str,
                 img_folder: str,
                 tension_threshold: int,
                 image_only: bool) -> None:
        """
        Initialize the TestTensionPredictions pipeline.

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
            encoder_path=None

        )
        self.tension_model = tf.keras.models.load_model(tension_model_path)
        self.image_only = image_only
        self.tester = TestPredictions(
            model_path=cnn_model_path, csv_path=csv_path,
            scalar_path=scaler_path, encoder_path=None,
            image_only=False,
            backbone="efficientnet",
            classification_type="binary",
            img_folder=img_folder
        )

        if self.classification_type == "multiclass":
            self.class_names = self.encoder.categories_[0].tolist()
        elif self.classification_type == "binary":
            self.class_names = [0, 1]
        self.scalar_path = scaler_path
        self.ohe = None
        if not self.image_only and self.scalar_path:
            self.feature_scaler = joblib.load(self.scalar_path)

    def predict_tension(self,
                        image_path:str,
                        params=None):
        image = self.load_process_images(image_path)
        image = np.expand_dims(image, axis=0)
        tension_pred = self.tension_model.predict([image, np.expand_dims(params, axis=0)])
        direction = np.argmax(tension_pred)
        return "Bad - raise tension" if direction == 1 else "Bad - lower tension"


    def gather_predictions(self, pred_features=None):
        """
        Gather predictions for image-only or image+feature based classification/regression.

        Args:
            pred_image_paths (list): Paths to the images for prediction.
            pred_features (np.ndarray, optional): Feature vectors for each image.

        Returns:
            Tuple of (true_labels, pred_labels)
        """
        pred_image_paths = self.df['ImagePath']

        pred_labels = []

        if self.image_only == False:
            for img_path in pred_image_paths:
                feature_vector = np.zeros((6,), dtype=np.float32)
                classification = self.tester.test_prediction(img_path, feature_vector)

                if classification < 0.5:
                    tension_pred = self.predict_tension(img_path, feature_vector)
                    pred_labels.append(tension_pred) 
        elif self.image_only:
            if pred_features is not None:
                for img_path, feature_vector in zip(pred_image_paths, pred_features):
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



from typing import Optional


class TensionPredictor:
    """
    Predicts tension values using a trained MLP model and preprocessed image/features.
    """

    def __init__(
        self,
        model_path: str,
        image_folder: str,
        csv_path: str,
        tension_scaler_path: Optional[str] = None,
        feature_scaler_path: Optional[str] = None,
    ):
        """
        Initialize TensionPredictor.

        Args:
            model (tf.keras.Model): Trained MLP model for tension prediction.
            image_folder (str): Path to image folder.
            image_path (str): Path to image for prediction.
            tension_scaler_path (str): Path to saved tension scaler.
            feature_scaler_path (str): Path to saved feature scaler.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.image_folder = image_folder
        if tension_scaler_path:
            self.tension_scaler = joblib.load(tension_scaler_path)
        if feature_scaler_path:
            self.feature_scaler = joblib.load(feature_scaler_path)
        self.csv_path = csv_path

    def load_and_preprocess_image(
        self, file_path: str, img_folder: str
    ) -> "tf.Tensor":
        """
        Load and preprocess image from file path.

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

    def PredictTension(self, features: "list[float]") -> float:
        """
        Predict tension for given image and features.

        Args:
            features (list[float]): Feature vector for prediction.

        Returns:
            float: Predicted tension value (inverse transformed to original scale).
        """
        image = self.load_and_preprocess_image(
            self.image_path, self.image_folder
        )
        image = tf.expand_dims(image, axis=0)
        features = np.array(features).reshape(1, -1)
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        # Predict tension
        features = self.feature_scaler.transform(features)
        predicted_tension = self.model.predict([image, features])
        # Scale tension back to normal units
        predicted_tension = self.tension_scaler.inverse_transform(
            predicted_tension
        )
        return predicted_tension[0][0]

    def find_best_tension_for_image(
        self,
        tension_range: List[float],
        other_features: Optional[np.ndarray] = None,
    ) -> None:
        """
        Find the tension that gives the best cleave quality prediction
        """
        df = pd.read_csv(self.csv_path)
        image_paths = df["ImagePath"]
        best_tensions = []
        for img in image_paths:
            img = self.load_and_preprocess_image(img, self.image_folder)
            img = tf.expand_dims(img, axis=0)
            best_tension = None
            best_prob = -1

            for tension in tension_range:
                if other_features is None:
                    features = np.zeros((6,))
                    features[1] = tension
                    features = np.expand_dims(features, axis=0)
                else:
                    features = other_features.copy()
                    features[1] = tension

                prediction = self.model.predict([img, features])
                quality_prob = prediction[0][0]

                if quality_prob > best_prob:
                    best_prob = quality_prob
                    best_tension = tension
                    best_tensions.append((best_tension, best_prob))

        for tension, prob in best_tensions:
            print(f"Best Tension: {tension:.2f}g -> Prob {prob:.2f}")

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
        """
        Plot a metric for model evaluation.

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
