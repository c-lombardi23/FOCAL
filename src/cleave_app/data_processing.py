"""Data processing module for the Fiber Cleave Processing application.

This module provides classes for loading, preprocessing, and organizing
data for training CNN and MLP models for fiber cleave analysis.
"""

import os
import warnings
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        train_test_split,
    )
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.applications.efficientnet import (
        preprocess_input as _efficientnet_preprocess,
    )
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input as _mobilenet_preprocess,
    )
    from tensorflow.keras.applications.resnet50 import (
        preprocess_input as _resnet_preprocess,
    )
except ImportError as e:
    print(f"Warning: Required ML libraries not found: {e}")
    print("Please install tensorflow>=2.19.0 and scikit-learn>=1.7.0")
    tf = None

# GLobal config variables
#==================================================================
IMAGE_DIMS = [224, 224]
IMAGE_SIZE = [224, 224, 3]
TRAIN_P=0.9
TEST_P=1.0
REQ_COLUMNS= [
                "CleaveAngle",
                "CleaveTension",
                "ScribeDiameter",
                "Misting",
                "Hackle",
                # "Tearing",
                "ImagePath",
                ]
FEATURES_CNN = [
            "CleaveAngle",
            "CleaveTension",
            "ScribeDiameter",
            "Misting",
            "Hackle",
            # "Tearing",
        ]
FEATURE_MLP = [
                "CleaveAngle",
                "ScribeDiameter",
                "Misting",
                "Hackle",
                # "Tearing",
            ]
#==================================================================

class DataCollector:
    """Class for collecting and preprocessing data from CSV files and image
    folders.

    This class handles loading cleave metadata from CSV files,
    processing images, and creating TensorFlow datasets for training
    machine learning models.
    """

    def __init__(
        self,
        csv_path: str,
        img_folder: str,
        angle_threshold: float,
        diameter_threshold: float,
        columns: List[str],
        classification_type: Optional[str] = "binary",
        backbone: Optional[str] = "mobilenet",
        set_mask: Optional[str] = "n",
        encoder_path: Optional[str] = None,
    ) -> None:
        """Initialize the data collector.

        Args:
            csv_path: Path to CSV file containing cleave metadata
            img_folder: Path to folder containing cleave images
            backbone: Optional pre-trained model to use as frozen layer
            classifcation_type: multiclass, multi_label, binary
        """
        if csv_path is None or img_folder is None:
            raise ValueError("Must provide both csv_path and img_folder")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        if not os.path.exists(img_folder):
            raise FileNotFoundError(f"Image folder not found: {img_folder}")

        self.csv_path = csv_path
        self.img_folder = img_folder
        self.feature_scaler = None
        self.label_scaler = None
        self.columns = columns
        self.encoder = None
        self.encoder_path = encoder_path
        self.classification_type = classification_type
        self._df = None
        self.backbone = backbone
        self.set_mask = set_mask
        self.angle_threshold = angle_threshold
        self.diameter_threshold = diameter_threshold
        

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """Lazy loading for memory efficiency."""
        if self._df is None:
            try:
                df = pd.read_csv(self.csv_path)
                required_columns = REQ_COLUMNS
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]
                if missing_columns:
                    raise ValueError(
                        f"CSV missing required columns: {missing_columns}"
                    )
            except pd.errors.EmptyDataError:
                raise ValueError(f"CSV file is empty: {self.csv_path}")
            except pd.errors.ParserError as e:
                raise ValueError(f"Invalid CSV format: {e}")

            self._df = self._clean_data()
        return self._df

    def _set_label(
        self, angle_threshold: float, diameter_threshold: float
    ) -> Optional[pd.DataFrame]:
        """Read CSV file and add cleave quality labels based on certain
        criteria.

        Returns:
            pd.DataFrame: DataFrame with added CleaveCategory column
        """
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"CSV file not found: {self.csv_path}")
            return None
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

        def label(row):
            good_angle = row["CleaveAngle"] <= angle_threshold
            no_defects = not row["Hackle"] and not row["Misting"]
            good_diameter = (
                row["ScribeDiameter"] < diameter_threshold * row["Diameter"]
            )

            bad_angle = not good_angle and no_defects and good_diameter
            bad_diameter = good_angle and no_defects and not good_diameter

            if good_angle and no_defects and good_diameter:
                return "Good"
            if good_angle and not no_defects and good_diameter:
                return "Misting_Hackle"
            if bad_angle or bad_diameter:
                return "Bad_Scribe_Mark or Angle"
            return "Multiple_Errors"

        if self.classification_type == "multiclass":
            df["CleaveCategory"] = df.apply(label, axis=1)
            return df
        if self.classification_type == "binary":

            df["CleaveCategory"] = df.apply(
                lambda row: (
                    1
                    if row["CleaveAngle"] <= angle_threshold
                    and row["ScribeDiameter"]
                    < diameter_threshold * row["Diameter"]
                    and (not row["Hackle"] and not row["Misting"])
                    else 0
                ),
                axis=1,
            )
            print(df["CleaveCategory"].value_counts())
            return df
        else:
            raise ValueError(f"Unsupported Mode {self.classification_type}")

    def save_scaler_encoder(self, obj: object, filepath: str) -> None:
        """Save a scaler or encoder to disk for future use.

        Args:
          filepath: Path to save scaler or encoder
          obj: Scaler or Encoder object
        """
        if not filepath.endswith(".pkl"):
            filepath = filepath + ".pkl"

        if not os.path.exists(filepath):
            joblib.dump(obj, filepath)
        else:
            raise FileExistsError("File Already exists!")

    def _clean_data(self) -> Optional[pd.DataFrame]:
        """Read CSV file and prepare data with cleave quality labels and one-
        hot encoding.

        Returns:
            pd.DataFrame: Processed DataFrame with labels and one-hot encoding
        """
        df = self._set_label(
            angle_threshold=self.angle_threshold,
            diameter_threshold=self.diameter_threshold,
        )
        if df is None:
            return None
        if self.classification_type == "multiclass":
            ohe = OneHotEncoder(sparse_output=False)
            onehot_labels = ohe.fit_transform(df[["CleaveCategory"]])
            class_names = ohe.categories_[0]

            for idx, class_name in enumerate(class_names):
                df[f"Label_{class_name}"] = onehot_labels[:, idx]

            self.encoder = ohe
            if self.encoder_path is not None:
                self.save_scaler_encoder(ohe, self.encoder_path)
                print(f"Encoder saved to {self.encoder_path}")

        # Clean image path by removing the base folder path
        df["ImagePath"] = df["ImagePath"].str.replace(
            f"{self.img_folder}\\", "", regex=False
        )
        return df

    def _mask_background(self, img: tf.Tensor) -> tf.Tensor:
        """Mask background to prevent model from focusing on sharp gradient
        near edges.

        Args:
            img: Image tensor of shape (H, W, C)

        Returns:
            tf.Tensor: Image with circular mask applied
        """
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        y_range = tf.range(h)
        x_range = tf.range(w)
        yy, xx = tf.meshgrid(y_range, x_range, indexing="ij")
        center_x = tf.cast(w, tf.float32) / 2.0
        center_y = tf.cast(h, tf.float32) / 2.0
        radius = tf.minimum(center_x, center_y)
        dist_from_center = tf.sqrt(
            (tf.cast(xx, tf.float32) - center_x) ** 2
            + (tf.cast(yy, tf.float32) - center_y) ** 2
        )

        mask = tf.cast(dist_from_center <= radius, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        return img * mask

    def get_backbone_preprocessor(self, backbone: str):
        """Return the preprocessing function for the specified backbone model.

        Args:
            backbone (str): Name of the backbone to use. Must be one of:
                - "mobilenet"
                - "resnet"
                - "efficientnet"

        Returns:
            Callable: The `preprocess_input` function tied to the chosen backbone.

        Raises:
            ValueError: If `backbone` is not one of the supported options.
        """
        mapping = {
            "mobilenet": _mobilenet_preprocess,
            "resnet": _resnet_preprocess,
            "efficientnet": _efficientnet_preprocess,
        }
        try:
            return mapping[backbone]
        except KeyError:
            raise ValueError(
                f"Invalid backbone: {backbone}.  Supported: {', '.join(mapping)}"
            )

    def load_process_images(self, filename: str) -> "tf.Tensor":
        """Load and preprocess image from file path.

        Args:
            filename: Image filename or path

        Returns:
            tf.Tensor: Preprocessed image tensor
        """

        backbone_preprocess = self.get_backbone_preprocessor(
            self.backbone or "efficientnet"
        )

        if tf is None:
            raise ImportError("TensorFlow is required for image processing")

        def _load_image(file, preprocess_input):
            """Load an image and process using same preprocessing as backbone.

            Args:
                file: path to image
                preprocess_input: processing from backbone model

            Returns:
                loaded and resized image
            """
            file = file.numpy().decode("utf-8")
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
                img = tf.image.resize(img, IMAGE_DIMS)
                img = tf.image.grayscale_to_rgb(img)
                if self.set_mask == "y":
                    img = self._mask_background(img)
                img = preprocess_input(img)
                return img
            except Exception as e:
                print(f"Error processing image {full_path}: {e}")
                return None

        img = tf.py_function(
            func=lambda f: _load_image(f, backbone_preprocess),
            inp=[filename],
            Tout=tf.float32,
        )
        img.set_shape(IMAGE_SIZE)
        return img

    def create_custom_dataset(
        self,
        image_shape: Tuple[int, int, int],
        test_size: float = 0.2,
        buffer_size: int = 32,
        batch_size: int = 16,
    ) -> Tuple["tf.data.Dataset", "tf.data.Dataset"]:
        """Create datasets using only grayscale images and labels with a custom
        image shape.

        Args:
            image_shape: Desired image shape (height, width, channels)
            test_size: Fraction of data to use for testing
            buffer_size: Buffer size for shuffling
            batch_size: Batch size for training

        Returns:
            Tuple of (train_ds, test_ds)
        """
        if tf is None:
            raise ImportError("TensorFlow is required for dataset creation")
        images = self.df["ImagePath"].values
        if self.classification_type == "multiclass":
            label_cols = [
                col for col in self.df.columns if col.startswith("Label_")
            ]
            labels = self.df[label_cols].values.astype(np.float32)
            stratify = np.argmax(labels, axis=1)
        elif self.classification_type == "binary":
            labels = self.df["CleaveCategory"].values
            stratify = labels

        train_imgs, test_imgs, train_labels, test_labels = train_test_split(
            images,
            labels,
            stratify=stratify,
            test_size=test_size,
            random_state=42,
        )

        def _load_grayscale_image(filename: str):
            """Load image in one color channel.

            Args:
                filename: filepath for mimage

            Raises:
                ValueError: if image cannot be converted

            Returns:
                loaded image
            """
            file = filename.numpy().decode("utf-8")
            full_path = os.path.join(self.img_folder, file)
            try:
                img_raw = tf.io.read_file(full_path)
                img = tf.image.decode_png(img_raw, channels=1)
                img = tf.image.resize(img, image_shape[:2])
                if self.set_mask == "y":
                    img = self._mask_background(img)
                img = tf.cast(img, tf.float32) / 255.0

                # Convert grayscale to RGB if needed
                if image_shape[2] == 3:
                    img = tf.image.grayscale_to_rgb(img)
                elif image_shape[2] != 1:
                    raise ValueError(
                        f"Unsupported number of channels: {image_shape[2]}"
                    )

                return img
            except Exception as e:
                print(f"Failed to load image {full_path}: {e}")
                return tf.zeros(image_shape, dtype=tf.float32)

        def process_fn(filename, label):
            img = tf.py_function(_load_grayscale_image, [filename], tf.float32)
            img.set_shape(image_shape)
            return img, label

        train_ds = tf.data.Dataset.from_tensor_slices(
            (train_imgs, train_labels)
        ).map(process_fn)
        test_ds = tf.data.Dataset.from_tensor_slices(
            (test_imgs, test_labels)
        ).map(process_fn)

        train_ds = (
            train_ds.shuffle(buffer_size=buffer_size)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, test_ds

    def extract_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract data from DataFrame into separate arrays for model training.

        Returns:
            Tuple of (images, features, labels) arrays
        """
        images = self.df["ImagePath"].values
        features = self.df[
            FEATURES_CNN
        ].values.astype(np.float32)
        labels = self.df["CleaveCategory"].values.astype(np.float32)

        return images, features, labels

    @staticmethod
    def _mask_features(images, features, p=0.3):
        """Randomly mask features to prevent reliance on numerical data."""

        def mask():
            return tf.zeros_like(features)

        features = tf.cond(tf.random.uniform([]) < p, mask, lambda: features)
        return (images, features)

    def _process_images_features(
        self, inputs: Tuple, label: np.ndarray
    ) -> Tuple[Tuple, np.ndarray]:
        """Process image and feature inputs for dataset creation.

        Args:
            inputs: Tuple of (image_input, features)
            label: Target label

        Returns:
            Tuple of processed inputs and label
        """
        image_input, features = inputs
        image = self.load_process_images(image_input)
        return (image, features), label

    def _dataset_helper(
        self,
        imgs: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        train: bool,
        batch_size: int,
        buffer_size: int,
        masking: bool,
        p: Optional[float] = None,
    ) -> tf.data.Dataset:
        """Helper function to create datasets from tensor slices and map image
        processing to each element.

        Args:
            imgs: Array of images paths
            features: Array of numerical features for the images
            labels: Target output (CleaveCategory in this case)
            train: Whether is train set or test set
            batch_size: Set the batch size to use during each training run
            buffer_size: Sets the size of the random buffer to introduce shuffling of data
            masking: Whether to use masking of feature array in training
            p: Probability of masking dataset

        Returns:
            tf.data.Dataset
        """
        ds = tf.data.Dataset.from_tensor_slices(((imgs, features), labels))
        ds = ds.map(lambda x, y: self._process_images_features(x, y))

        if masking:
            if p is not None:
                ds = ds.map(
                    lambda x, y: (
                        DataCollector._mask_features(x[0], x[1], p=p),
                        y,
                    )
                )
            else:
                raise ValueError("P value cannot be None!")
        if train:
            ds = (
                ds.shuffle(buffer_size=buffer_size)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def create_kfold_datasets(
        self,
        images: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        buffer_size: int,
        batch_size: int,
        n_splits: int = 5,
    ) -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
        """Create datasets based on stratified k-fold cross validation.

        Args:
            images: Array of image paths
            features: Array of numerical features
            labels: Array of target labels
            buffer_size: Buffer size for dataset shuffling
            batch_size: Batch size for training
            n_splits: Number of k-fold splits

        Returns:
            List of (train_ds, test_ds) tuples for each fold
        """
        if tf is None:
            raise ImportError("TensorFlow is required for dataset creation")

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=24)
        datasets = []
        if self.classification_type == "multiclass":
            single_labels = np.argmax(labels, axis=1)
        elif self.classification_type == "binary":
            single_labels = labels

        for train_index, test_index in kf.split(X=features, y=single_labels):
            train_imgs, test_imgs = (
                images[train_index],
                images[test_index],
            )
            train_features, test_features = (
                features[train_index],
                features[test_index],
            )
            train_labels, test_labels = (
                labels[train_index],
                labels[test_index],
            )

            train_ds = self._dataset_helper(
                train_imgs,
                train_features,
                train_labels,
                train=True,
                batch_size=batch_size,
                buffer_size=buffer_size,
                masking=True,
                p=TRAIN_P,
            )
            test_ds = self._dataset_helper(
                test_imgs,
                test_features,
                test_labels,
                train=False,
                batch_size=batch_size,
                buffer_size=buffer_size,
                masking=True,
                p=TEST_P,
            )

            datasets.append((train_ds, test_ds))

        return datasets

    def create_datasets(
        self,
        images: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float,
        buffer_size: int,
        batch_size: int,
        feature_scaler_path: Optional[str] = None,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[dict[int, float]]]:
        """Create train and test datasets with feature scaling.

        Args:
            images: Array of image paths
            features: Array of numerical features
            labels: Array of target labels
            test_size: Fraction of data to use for testing
            buffer_size: Buffer size for dataset shuffling
            batch_size: Batch size for training
            feature_scaler_path: Optional path to save feature scaler
            encoder_path: Optional path to save one-hot-encoder

        Returns:
            Tuple of (train_ds, test_ds)
        """
        if tf is None:
            raise ImportError("TensorFlow is required for dataset creation")

        # Stratified split for classification
        if self.classification_type == "binary":
            class_weights_array = compute_class_weight(
                class_weight="balanced",
                classes=np.array([0, 1]),
                y=labels,
            )
            class_weights = dict(enumerate(class_weights_array))
            stratify = labels
        elif self.classification_type == "multiclass":
            stratify = labels.argmax(axis=1)
            class_weights = None

        (
            train_imgs,
            test_imgs,
            train_features,
            test_features,
            train_labels,
            test_labels,
        ) = train_test_split(
            images,
            features,
            labels,
            stratify=stratify,
            test_size=test_size,
            random_state=42,
        )

        # Scale features
        scaler = MinMaxScaler()
        self.feature_scaler = scaler
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        # Save scaler if path provided
        if feature_scaler_path:
            self.save_scaler_encoder(scaler, feature_scaler_path)
            print(f"Feature scaler saved to: {feature_scaler_path}")

        # Create datasets
        train_ds = self._dataset_helper(
            train_imgs,
            train_features,
            train_labels,
            train=True,
            batch_size=batch_size,
            buffer_size=buffer_size,
            masking=True,
            p=TRAIN_P,
        )
        test_ds = self._dataset_helper(
            test_imgs,
            test_features,
            test_labels,
            train=False,
            batch_size=batch_size,
            buffer_size=buffer_size,
            masking=False,
            p=TEST_P,
        )

        return train_ds, test_ds, class_weights

    def image_only_dataset(
        self, original_dataset: tf.data.Dataset
    ) -> tf.data.Dataset:
        """Convert dataset to image-only format (remove feature inputs).

        Args:
            original_dataset: Original dataset with (image, features) inputs

        Returns:
            tf.data.Dataset: Dataset with only image inputs
        """
        return original_dataset.map(lambda inputs, label: (inputs[0], label))


class MLPDataCollector(DataCollector):
    """Data collector specifically for MLP regression models.

    This class handles data preparation for tension prediction models,
    including proper scaling of both features and labels.
    """

    def __init__(
        self,
        csv_path: str,
        img_folder: str,
        angle_threshold: float,
        diameter_threshold: float,
        backbone: Optional[str] = None,
    ):
        """Initialize the MLP data collector.

        Args:
            csv_path: Path to CSV file containing cleave metadata
            img_folder: Path to folder containing cleave images
        """
        super().__init__(
            csv_path,
            img_folder,
            backbone=backbone,
            angle_threshold=angle_threshold,
            diameter_threshold=diameter_threshold,
        )

    def extract_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract data for MLP regression (tension prediction).

        Returns:
            Tuple of (images, features, labels) arrays
        """
        if self.df is None:
            raise ValueError(
                "No data available. Check if CSV file was loaded correctly."
            )
        filtered_df = self.df.loc[self.df["CleaveCategory"] == 1]
        # mean_tension = np.mean(filtered_df["CleaveTension"])
        mean_tension = (
            filtered_df.groupby("FiberType")["CleaveTension"].mean().to_dict()
        )

        self.df["MeanTension"] = self.df["FiberType"].map(mean_tension)
        delta = np.where(
            self.df["CleaveCategory"] == 1,
            0.0,
            self.df["MeanTension"] - self.df["CleaveTension"],
        ).astype(np.float32)

        images = self.df["ImagePath"].values
        features = self.df[
            FEATURE_MLP
        ].values.astype(np.float32)
        labels = delta

        return images, features, labels

    def create_datasets(
        self,
        images: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float,
        buffer_size: int,
        batch_size: int,
        feature_scaler_path: Optional[str] = None,
        tension_scaler_path: Optional[str] = None,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Create train and test datasets for MLP regression with proper
        scaling.

        Args:
            images: Array of image paths
            features: Array of numerical features
            labels: Array of tension values
            test_size: Fraction of data to use for testing
            buffer_size: Buffer size for dataset shuffling
            batch_size: Batch size for training
            feature_scaler_path: Optional path to save feature scaler
            tension_scaler_path: Optional path to save tension scaler

        Returns:
            Tuple of (train_ds, test_ds)
        """
        if tf is None:
            raise ImportError("TensorFlow is required for dataset creation")

        # Split data
        (
            train_imgs,
            test_imgs,
            train_features,
            test_features,
            train_labels,
            test_labels,
        ) = train_test_split(
            images,
            features,
            labels,
            test_size=test_size,
            random_state=42,
        )

        # Scale features
        feature_scaler = MinMaxScaler()
        self.feature_scaler = feature_scaler
        train_features = feature_scaler.fit_transform(train_features)
        test_features = feature_scaler.transform(test_features)

        # Scale labels
        tension_scaler = MinMaxScaler()
        self.label_scaler = tension_scaler
        train_labels = tension_scaler.fit_transform(
            train_labels.reshape(-1, 1)
        )
        test_labels = tension_scaler.transform(test_labels.reshape(-1, 1))

        # Save scalers if paths provided
        if feature_scaler_path:
            self.save_scaler_encoder(feature_scaler, feature_scaler_path)
            print(f"Feature scaler saved to: {feature_scaler_path}")

        if tension_scaler_path:
            self.save_scaler_encoder(tension_scaler, tension_scaler_path)
            print(f"Tension scaler saved to: {tension_scaler_path}")

        # Create datasets
        train_ds = self._dataset_helper(
            train_imgs,
            train_features,
            train_labels,
            train=True,
            batch_size=batch_size,
            buffer_size=buffer_size,
            masking=False,
        )
        test_ds = self._dataset_helper(
            test_imgs,
            test_features,
            test_labels,
            train=False,
            batch_size=batch_size,
            buffer_size=buffer_size,
            masking=False,
        )

        return train_ds, test_ds

    def create_kfold_datasets(
        self,
        images: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        buffer_size: int,
        batch_size: int,
        n_splits: int = 5,
    ) -> Tuple[List[Tuple[tf.data.Dataset, tf.data.Dataset]], MinMaxScaler]:
        """Create k-fold datasets for MLP regression with proper scaling.

        Args:
            images: Array of image paths
            features: Array of numerical features
            labels: Array of tension values
            buffer_size: Buffer size for dataset shuffling
            batch_size: Batch size for training
            n_splits: Number of k-fold splits

        Returns:
            Tuple of (datasets, label_scaler)
        """
        if tf is None:
            raise ImportError("TensorFlow is required for dataset creation")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=24)
        datasets = []

        # Scale features and labels globally
        feature_scaler = MinMaxScaler()
        label_scaler = MinMaxScaler()

        scaled_features = feature_scaler.fit_transform(features)
        scaled_labels = label_scaler.fit_transform(labels.reshape(-1, 1))

        for train_index, test_index in kf.split(images):
            train_imgs, test_imgs = (
                images[train_index],
                images[test_index],
            )
            train_features, test_features = (
                scaled_features[train_index],
                scaled_features[test_index],
            )
            train_labels, test_labels = (
                scaled_labels[train_index],
                scaled_labels[test_index],
            )

            train_ds = self._dataset_helper(
                train_imgs,
                train_features,
                train_labels,
                train=True,
                batch_size=batch_size,
                buffer_size=buffer_size,
                masking=False,
            )
            test_ds = self._dataset_helper(
                test_imgs,
                test_features,
                test_labels,
                train=False,
                batch_size=batch_size,
                buffer_size=buffer_size,
                masking=False,
            )

            datasets.append((train_ds, test_ds))

        return datasets, label_scaler


class BadCleaveTensionClassifier(DataCollector):
    def __init__(
        self,
        csv_path: str,
        img_folder: str,
        tension_threshold: int,
        backbone: Optional[str] = "efficientnet",
        encoder_path: Optional[str] = None,
        classification_type: Optional[str] = "binary",
    ):

        self.tension_threshold = tension_threshold
        super().__init__(
            csv_path=csv_path,
            img_folder=img_folder,
            classification_type="binary",
            backbone="efficientnet",
            encoder_path=encoder_path,
        )

    def _clean_data(self):
        df = super()._clean_data()
        df = df.loc[df["CleaveCategory"] == 0]
        df["BadTensionsLabel"] = (
            df["CleaveTension"] > self.tension_threshold
        ).astype(np.int32)
        print(df["BadTensionsLabel"].value_counts())
        return df

    def extract_data(self):
        images, features, labels = super().extract_data()
        labels = self.df["BadTensionsLabel"]

        return images, features, labels
