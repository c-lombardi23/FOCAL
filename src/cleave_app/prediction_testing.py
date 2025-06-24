#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder



class TestPredictions:
    """
    This class is used to test model performance on unseen data using metrics such as
    accuracy, precision, recall, and confusion matrix. Supports both image+feature and image-only CNNs.
    """
    def __init__(self, model_path: str, csv_path: str, scalar_path: str, img_folder: str, image_only: bool = False, backbone: str = "mobilenet"):
        """
        Initialize TestPredictions.

        Args:
            model_path (str): Path to trained model.
            csv_path (str): Path to CSV file with metadata.
            scalar_path (str): Path to feature scaler (ignored if image_only=True).
            img_folder (str): Path to image folder.
            image_only (bool): If True, test only with images (no features).
        """
        self.scalar_path = scalar_path
        self.img_folder = img_folder
        self.model = tf.keras.models.load_model(model_path)
        self.csv_path = csv_path
        self.image_only = image_only
        self.df = self.clean_data()
        self.scaler = None
        self.backbone = backbone
        if not self.image_only and self.scalar_path:
            self.scaler = joblib.load(self.scalar_path)

    def set_label(self) -> 'pd.DataFrame | None':
        """
        Read CSV file and add cleave quality labels based on criteria.

        Returns:
            pd.DataFrame | None: DataFrame with added CleaveCategory column, or None if file not found.
        """
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print("CSV file not found!")
            return None
        def label(row):
            good_angle = row['CleaveAngle'] <= 0.45
            no_defects = not row['Hackle'] and not row['Misting']
            good_diameter = row['ScribeDiameter'] < 17
            if good_angle and no_defects and good_diameter:
                return "Good"
            elif (good_angle and not no_defects and good_diameter):
                return "Misting_Hackle"
            elif (good_angle and no_defects and not good_diameter):
                return "Bad_Scribe_Mark"
            elif (not good_angle and no_defects and good_diameter):
                return "Bad_Angle"
            else:
                return "Multiple_Errors"
        df["CleaveCategory"] = df.apply(label, axis=1)
        return df

    def clean_data(self) -> 'pd.DataFrame | None':
        """
        Read CSV file into DataFrame and add column for cleave quality and one-hot encoded labels.

        Returns:
            pd.DataFrame | None: DataFrame with cleave quality and one-hot labels, or None if file not found.
        """
        try:
            df = self.set_label()
        except FileNotFoundError:
            print("CSV file not found!")
            return None
        # Clean image path
        df['ImagePath'] = df['ImagePath'].str.replace(self.img_folder, "", regex=False)
        # One-hot encode CleaveCategory
        ohe = OneHotEncoder()
        onehot_labels = ohe.fit_transform(df[['CleaveCategory']]).toarray()
        class_names = ohe.categories_[0]
        for idx, class_name in enumerate(class_names):
            df[f"Label_{class_name}"] = onehot_labels[:, idx]
        self.encoder = ohe
        self.class_names = class_names
        return df

    def load_process_images(self, filename) -> tf.Tensor:
        """
        Load and preprocess image from file path.

        Args:
            filename: Image filename or path

        Returns:
            tf.Tensor: Preprocessed image tensor
        """
        if self.backbone == "mobilenet":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as backbone_preprocess
        elif self.backbone == "resnet":
            from tensorflow.keras.applications.resnet50 import preprocess_input as backbone_preprocess
        elif self.backbone == "efficientnet":
            from tensorflow.keras.applications.efficientnet import preprocess_input as backbone_preprocess
        else:
            raise ValueError(f"Invalid backbone: {self.backbone}")

        if tf is None:
            raise ImportError("TensorFlow is required for image processing")
            
        def _load_image(file, preprocess_input):
            """Load and preprocess a single image."""
            file = file.numpy().decode('utf-8')
            full_path = os.path.join(self.img_folder, file)
            
            try:
                img_raw = tf.io.read_file(full_path)
            except FileNotFoundError:
                print(f"Image file not found: {full_path}")
                return None
            except Exception as e:
                print(f"Error loading image {full_path}: {e}")
                return None
                
            try:
                img = tf.image.decode_png(img_raw, channels=1)
                img = tf.image.resize(img, [224, 224])
                img = tf.image.grayscale_to_rgb(img)
                img = preprocess_input(img)
                return img
            except Exception as e:
                print(f"Error processing image {full_path}: {e}")
                return None

        img = tf.py_function(
        func=lambda f: _load_image(f, backbone_preprocess),
        inp=[filename],
        Tout=tf.float32
    )
        img.set_shape([224, 224, 3])
        return img


    def test_prediction(self, image_path: str, feature_vector: 'np.ndarray | None' = None) -> 'np.ndarray':
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
            prediction = self.model.predict([image, feature_vector])
        return prediction

    def gather_predictions(self) -> 'tuple[np.ndarray, list, list] | tuple[None, None, None]':
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
            pred_features = self.df[['CleaveAngle', 'CleaveTension', 'ScribeDiameter', 'Misting', 'Hackle', 'Tearing']].values
            if self.scaler is not None:
                pred_features = self.scaler.transform(pred_features)
        predictions = []
        if self.image_only:
            for img_path in pred_image_paths:
                prediction = self.test_prediction(img_path)
                predictions.append(prediction)
        else:
            if pred_features is not None:
                for img_path, feature_vector in zip(pred_image_paths, pred_features):
                    prediction = self.test_prediction(img_path, feature_vector)
                    predictions.append(prediction)
            else:
                print("No features available for prediction.")
                return None, None, None
        # Set prediction labels based on max of ohe
        pred_labels = [np.argmax(pred[0]) for pred in predictions]
        true_labels = self.df["CleaveCategory"].map({label: idx for idx, label in enumerate(self.class_names)}).values
        return true_labels, pred_labels, predictions

    def display_confusion_matrix(self, true_labels: 'np.ndarray', pred_labels: 'list[int]') -> None:
        """
        Display confusion matrix comparing true labels to predicted labels.

        Args:
            true_labels (np.ndarray): Array of true labels.
            pred_labels (list[int]): List of predicted labels.
        """
        labels = list(range(len(self.class_names)))
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=self.class_names)
        disp.plot()
        plt.show()

    def display_classification_report(self, true_labels: 'np.ndarray', pred_labels: 'list[int]', classification_path: str = None) -> None:
        """
        Display classification report comparing true labels to predicted labels.

        Args:
            true_labels (np.ndarray): Array of true labels.
            pred_labels (list[int]): List of predicted labels.
            classification_path (str, optional): Optional path to save classification report.
        """
        if classification_path:
            df = pd.DataFrame(classification_report(true_labels, pred_labels, target_names=self.class_names, output_dict=True)).transpose()
            df.to_csv(classification_path, index=True)
        print(classification_report(true_labels, pred_labels, target_names=self.class_names))

    def plot_roc(self, title: str, true_labels: 'np.ndarray', pred_probabilites: 'np.ndarray') -> None:
        """
        Plot ROC curve for predictions.

        Args:
            title (str): Title for the plot.
            true_labels (np.ndarray): Array of true labels.
            pred_probabilites (np.ndarray): Array of predicted probabilities.
        """
        pred_probabilites = np.array(pred_probabilites).flatten()
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probabilites)
        auc = roc_auc_score(true_labels, pred_probabilites)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc:.2f}%)')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

class TensionPredictor:
    """
    Predicts tension values using a trained MLP model and preprocessed image/features.
    """
    def __init__(self, model: 'tf.keras.Model', image_folder: str, image_path: str, tension_scaler_path: str, feature_scaler_path: str):
        """
        Initialize TensionPredictor.

        Args:
            model (tf.keras.Model): Trained MLP model for tension prediction.
            image_folder (str): Path to image folder.
            image_path (str): Path to image for prediction.
            tension_scaler_path (str): Path to saved tension scaler.
            feature_scaler_path (str): Path to saved feature scaler.
        """
        self.model = model
        self.image_path = image_path
        self.image_folder = image_folder
        self.tension_scaler = joblib.load(tension_scaler_path)
        self.feature_scaler = joblib.load(feature_scaler_path)

    def load_and_preprocess_image(self, file_path: str, img_folder: str) -> 'tf.Tensor':
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
        img = img / 255.0
        return img

    def PredictTension(self, features: 'list[float]') -> float:
        """
        Predict tension for given image and features.

        Args:
            features (list[float]): Feature vector for prediction.

        Returns:
            float: Predicted tension value (inverse transformed to original scale).
        """
        image = self.load_and_preprocess_image(self.image_path, self.image_folder)
        image = tf.expand_dims(image, axis=0)
        features = np.array(features).reshape(1, -1)
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        # Predict tension
        features = self.feature_scaler.transform(features)
        predicted_tension = self.model.predict([image, features])
        # Scale tension back to normal units
        predicted_tension = self.tension_scaler.inverse_transform(predicted_tension)
        return predicted_tension[0][0]

    def plot_metric(self, title: str, X: 'list[float]', y: 'list[float]', x_label: str, y_label: str, x_legend: str, y_legend: str) -> None:
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
        plt.legend(loc='lower right')
        plt.show()