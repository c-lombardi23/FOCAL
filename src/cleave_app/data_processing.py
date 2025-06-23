"""
Data processing module for the Fiber Cleave Processing application.

This module provides classes for loading, preprocessing, and organizing data
for training CNN and MLP models for fiber cleave analysis.
"""

import os
import warnings
from typing import Optional, Tuple, List, Any, Union
import pandas as pd
import numpy as np
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.utils.class_weight import compute_class_weight
except ImportError as e:
    print(f"Warning: Required ML libraries not found: {e}")
    print("Please install tensorflow>=2.19.0 and scikit-learn>=1.7.0")
    tf = None


class DataCollector:
    """
    Class for collecting and preprocessing data from CSV files and image folders.
    
    This class handles loading cleave metadata from CSV files, processing images,
    and creating TensorFlow datasets for training machine learning models.
    """
    
    def __init__(self, csv_path: str, img_folder: str):
        """
        Initialize the data collector.

        Args:
            csv_path: Path to CSV file containing cleave metadata
            img_folder: Path to folder containing cleave images
        """
        if csv_path is None or img_folder is None:
            raise ValueError("Must provide both csv_path and img_folder")
            
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        if not os.path.exists(img_folder):
            raise FileNotFoundError(f"Image folder not found: {img_folder}")
            
        self.csv_path = csv_path
        self.img_folder = img_folder
        self.df = self.clean_data()
        self.feature_scaler = None
        self.label_scaler = None
        self.encoder = None

    def set_label(self) -> Optional[pd.DataFrame]:
        """
        Read CSV file and add cleave quality labels based on criteria.

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
            """Label cleave quality based on angle, defects, and diameter."""
            good_angle = row['CleaveAngle'] <= 0.45
            no_defects = not row['Hackle'] and not row['Misting']
            good_diameter = row['ScribeDiameter'] < 17
            
            if good_angle and no_defects and good_diameter:
                return "Good"
            elif good_angle and not no_defects and good_diameter:
                return "Misting_Hackle"
            elif good_angle and no_defects and not good_diameter:
                return "Bad_Scribe_Mark"
            elif not good_angle and no_defects and good_diameter:
                return "Bad_Angle"
            else:
                return "Multiple_Errors"

        df["CleaveCategory"] = df.apply(label, axis=1)
        return df

    def clean_data(self) -> Optional[pd.DataFrame]:
        """
        Read CSV file and prepare data with cleave quality labels and one-hot encoding.

        Returns:
            pd.DataFrame: Processed DataFrame with labels and one-hot encoding
        """
        df = self.set_label()
        if df is None:
            return None

        # Clean image path by removing the base folder path
        df['ImagePath'] = df['ImagePath'].str.replace(self.img_folder, "", regex=False)

        # One-hot encode CleaveCategory
        ohe = OneHotEncoder(sparse_output=False)
        onehot_labels = ohe.fit_transform(df[['CleaveCategory']])
        class_names = ohe.categories_[0]

        for idx, class_name in enumerate(class_names):
            df[f"Label_{class_name}"] = onehot_labels[:, idx]

        self.encoder = ohe
        return df

    def load_process_images(self, filename) -> tf.Tensor:
        """
        Load and preprocess image from file path.

        Args:
            filename: Image filename or path

        Returns:
            tf.Tensor: Preprocessed image tensor
        """
        if tf is None:
            raise ImportError("TensorFlow is required for image processing")
            
        def _load_image(file):
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
                img = img / 255.0
                return img
            except Exception as e:
                print(f"Error processing image {full_path}: {e}")
                return None

        img = tf.py_function(_load_image, [filename], tf.float32)
        img.set_shape([224, 224, 3])
        return img

    def extract_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract data from DataFrame into separate arrays for model training.

        Returns:
            Tuple of (images, features, labels) arrays
        """
        if self.df is None:
            raise ValueError("No data available. Check if CSV file was loaded correctly.")
            
        images = self.df['ImagePath'].values
        features = self.df[['CleaveAngle', 'CleaveTension', 'ScribeDiameter', 'Misting', 'Hackle', 'Tearing']].values.astype(np.float32)
        label_cols = [col for col in self.df.columns if col.startswith('Label_')]
        labels = self.df[label_cols].values.astype(np.float32)
        
        return images, features, labels

    def process_images_features(self, inputs: Tuple, label: np.ndarray) -> Tuple[Tuple, np.ndarray]:
        """
        Process image and feature inputs for dataset creation.

        Args:
            inputs: Tuple of (image_input, features)
            label: Target label

        Returns:
            Tuple of processed inputs and label
        """
        image_input, features = inputs
        image = self.load_process_images(image_input)
        return (image, features), label

    def create_kfold_datasets(self, 
                            images: np.ndarray, 
                            features: np.ndarray, 
                            labels: np.ndarray, 
                            buffer_size: int, 
                            batch_size: int, 
                            n_splits: int = 5) -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
        """
        Create datasets based on stratified k-fold cross validation.

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
        single_labels = np.argmax(labels, axis=1)

        for train_index, test_index in kf.split(X=features, y=single_labels):
            train_imgs, test_imgs = images[train_index], images[test_index]
            train_features, test_features = features[train_index], features[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]

            train_ds = tf.data.Dataset.from_tensor_slices(((train_imgs, train_features), train_labels))
            test_ds = tf.data.Dataset.from_tensor_slices(((test_imgs, test_features), test_labels))

            train_ds = train_ds.map(lambda x, y: self.process_images_features(x, y))
            test_ds = test_ds.map(lambda x, y: self.process_images_features(x, y))

            train_ds = train_ds.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            datasets.append((train_ds, test_ds))

        return datasets

    def create_datasets(self, 
                       images: np.ndarray, 
                       features: np.ndarray, 
                       labels: np.ndarray, 
                       test_size: float, 
                       buffer_size: int, 
                       batch_size: int, 
                       feature_scaler_path: Optional[str] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create train and test datasets with feature scaling.

        Args:
            images: Array of image paths
            features: Array of numerical features
            labels: Array of target labels
            test_size: Fraction of data to use for testing
            buffer_size: Buffer size for dataset shuffling
            batch_size: Batch size for training
            feature_scaler_path: Optional path to save feature scaler

        Returns:
            Tuple of (train_ds, test_ds)
        """
        if tf is None:
            raise ImportError("TensorFlow is required for dataset creation")
            
        # Stratified split for classification
        stratify_labels = labels.argmax(axis=1)
        train_imgs, test_imgs, train_features, test_features, train_labels, test_labels = train_test_split(
            images, features, labels, stratify=stratify_labels, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = MinMaxScaler()
        self.feature_scaler = scaler
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        
        # Save scaler if path provided
        if feature_scaler_path:
            os.makedirs(os.path.dirname(feature_scaler_path), exist_ok=True)
            if not feature_scaler_path.endswith(".pkl"):
                feature_scaler_path = feature_scaler_path + ".pkl"
            joblib.dump(scaler, feature_scaler_path)
            print(f"Feature scaler saved to: {feature_scaler_path}")
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices(((train_imgs, train_features), train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices(((test_imgs, test_features), test_labels))

        train_ds = train_ds.map(lambda x, y: self.process_images_features(x, y))
        test_ds = test_ds.map(lambda x, y: self.process_images_features(x, y))

        train_ds = train_ds.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, test_ds

    def image_only_dataset(self, original_dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Convert dataset to image-only format (remove feature inputs).

        Args:
            original_dataset: Original dataset with (image, features) inputs

        Returns:
            tf.data.Dataset: Dataset with only image inputs
        """
        return original_dataset.map(lambda inputs, label: (inputs[0], label))


class MLPDataCollector(DataCollector):
    """
    Data collector specifically for MLP regression models.
    
    This class handles data preparation for tension prediction models,
    including proper scaling of both features and labels.
    """

    def __init__(self, csv_path: str, img_folder: str):
        """
        Initialize the MLP data collector.

        Args:
            csv_path: Path to CSV file containing cleave metadata
            img_folder: Path to folder containing cleave images
        """
        super().__init__(csv_path, img_folder)

    def extract_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract data for MLP regression (tension prediction).

        Returns:
            Tuple of (images, features, labels) arrays
        """
        if self.df is None:
            raise ValueError("No data available. Check if CSV file was loaded correctly.")
            
        images = self.df['ImagePath'].values
        features = self.df[['CleaveAngle', 'ScribeDiameter', 'Misting', 'Hackle', 'Tearing']].values.astype(np.float32)
        labels = self.df['CleaveTension'].values.astype(np.float32)
        
        return images, features, labels

    def create_datasets(self, 
                       images: np.ndarray, 
                       features: np.ndarray, 
                       labels: np.ndarray, 
                       test_size: float, 
                       buffer_size: int, 
                       batch_size: int, 
                       feature_scaler_path: Optional[str] = None, 
                       tension_scaler_path: Optional[str] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create train and test datasets for MLP regression with proper scaling.

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
            
        # Split data (no stratification for regression)
        train_imgs, test_imgs, train_features, test_features, train_labels, test_labels = train_test_split(
            images, features, labels, test_size=test_size, random_state=42
        )
        
        # Scale features
        feature_scaler = MinMaxScaler()
        self.feature_scaler = feature_scaler
        train_features = feature_scaler.fit_transform(train_features)
        test_features = feature_scaler.transform(test_features)
        
        # Scale labels (tension values)
        tension_scaler = MinMaxScaler()
        self.label_scaler = tension_scaler
        train_labels = tension_scaler.fit_transform(train_labels.reshape(-1, 1))
        test_labels = tension_scaler.transform(test_labels.reshape(-1, 1))

        # Save scalers if paths provided
        if feature_scaler_path:
            os.makedirs(os.path.dirname(feature_scaler_path), exist_ok=True)
            if not feature_scaler_path.endswith(".pkl"):
                feature_scaler_path = feature_scaler_path + ".pkl"
            joblib.dump(feature_scaler, feature_scaler_path)
            print(f"Feature scaler saved to: {feature_scaler_path}")
            
        if tension_scaler_path:
            os.makedirs(os.path.dirname(tension_scaler_path), exist_ok=True)
            if not tension_scaler_path.endswith(".pkl"):
                tension_scaler_path = tension_scaler_path + ".pkl"
            joblib.dump(tension_scaler, tension_scaler_path)
            print(f"Tension scaler saved to: {tension_scaler_path}")

        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices(((train_imgs, train_features), train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices(((test_imgs, test_features), test_labels))

        train_ds = train_ds.map(lambda x, y: self.process_images_features(x, y))
        test_ds = test_ds.map(lambda x, y: self.process_images_features(x, y))

        train_ds = train_ds.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, test_ds

    def create_kfold_datasets(self, 
                            images: np.ndarray, 
                            features: np.ndarray, 
                            labels: np.ndarray, 
                            buffer_size: int, 
                            batch_size: int, 
                            n_splits: int = 5) -> Tuple[List[Tuple[tf.data.Dataset, tf.data.Dataset]], MinMaxScaler]:
        """
        Create k-fold datasets for MLP regression with proper scaling.

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
            train_imgs, test_imgs = images[train_index], images[test_index]
            train_features, test_features = scaled_features[train_index], scaled_features[test_index]
            train_labels, test_labels = scaled_labels[train_index], scaled_labels[test_index]

            train_ds = tf.data.Dataset.from_tensor_slices(((train_imgs, train_features), train_labels))
            test_ds = tf.data.Dataset.from_tensor_slices(((test_imgs, test_features), test_labels))

            train_ds = train_ds.map(lambda x, y: self.process_images_features(x, y))
            test_ds = test_ds.map(lambda x, y: self.process_images_features(x, y))

            train_ds = train_ds.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            datasets.append((train_ds, test_ds))
            
        return datasets, label_scaler