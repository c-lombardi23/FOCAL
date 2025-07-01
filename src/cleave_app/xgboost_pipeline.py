from typing import Optional
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
import xgboost as xgb
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from .base_model_pipeline import BaseModelPipeline

class XGBoostModel(BaseModelPipeline):

    def __init__(self, csv_path: str, cnn_model_path: str, train_ds, test_ds):
        self.csv_path = csv_path
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.xgb_reg = None

        try:
            self.cnn_model = tf.keras.models.load_model(cnn_model_path)
            self.feature_extractor = Model(
                inputs=self.cnn_model.input[0],
                outputs=self.cnn_model.get_layer("global_avg").output
            )
        except ValueError as e:
            print(f"Error loading model or extracting layer: {e}")
            self.feature_extractor = None

    def extract_features_and_labels(self, ds):
        features, tensions = [], []
        for img_batch, tension_batch in ds:
            feats = self.feature_extractor(img_batch).numpy()
            features.append(feats)
            tensions.append(tension_batch.numpy().reshape(-1))
        return np.vstack(features), np.concatenate(tensions)

    def train(self,
              n_estimators: Optional[int] = 200,
              learning_rate: Optional[float] = 0.05,
              max_depth: Optional[int] = 4,
              random_state: Optional[int] = 42):
        
        X_train, y_train = self.extract_features_and_labels(self.train_ds)
        X_test, y_test = self.extract_features_and_labels(self.test_ds)

        self.xgb_reg = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

        self.xgb_reg.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=True
        )

        from sklearn.metrics import mean_absolute_error
        y_pred = self.xgb_reg.predict(X_test)
        print("Val MAE:", mean_absolute_error(y_test, y_pred))

        return self.xgb_reg.evals_result()

    def save(self, save_path: str):
        if self.xgb_reg is not None:
            joblib.dump(self.xgb_reg, save_path)
        else:
            raise ValueError("Model not trained yet.")

    def load(self, model_path: str):
        self.xgb_reg = joblib.load(model_path)

    def predict(self, data):
        features = self.extract_features_and_labels(data)[0]
        return self.xgb_reg.predict(features)

    def plot_metrics(self, title: str, metric1, metric2, 
                     metric1_label: str, metric2_label: str,
                     x_label: str, y_label: str) -> None:
        plt.title(title)
        plt.plot(metric1, label=metric1_label)
        plt.plot(metric2, label=metric2_label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper right")
        plt.show()

class XGBoostPredictor(BaseModelPipeline):
    def __init__(
        self,
        csv_path: str,
        cnn_model_path: str,
        xgb_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
    ):
        self.csv_path = csv_path
        self.xgb_path = xgb_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None

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

    def extract_cnn_features(self, image_path: str) -> np.ndarray:
        """Extract CNN features from a grayscale image."""
        img_raw = tf.io.read_file(image_path)
        img = tf.image.decode_png(img_raw, channels=1)
        img = tf.image.resize(img, [224, 224])
        img = tf.image.grayscale_to_rgb(img)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)
        return self.feature_extractor(img).numpy().squeeze()

    def extract_data(self):
        """Load and filter dataset for prediction (only bad cleaves)."""
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV: {e}")

        df["CleaveCategory"] = df.apply(
            lambda row: 1
            if row["CleaveAngle"] <= 0.45
            and row["ScribeDiameter"] < 17
            and not row["Hackle"]
            and not row["Misting"]
            else 0,
            axis=1,
        )
        return df[df["CleaveCategory"] == 0]

    def load(self):
        """Load trained model and scaler."""
        if not self.xgb_path or not self.scaler_path:
            raise ValueError("Paths for model and scaler must be provided.")

        try:
            self.model = joblib.load(self.xgb_path)
            self.scaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or scaler: {e}")

    def predict(self):
        """Run tension predictions on filtered cleave data."""
        if not self.model or not self.scaler:
            raise RuntimeError("Model and scaler must be loaded before prediction.")

        df = self.extract_data()
        image_paths = df["ImagePath"]
        true_tensions = df["CleaveTension"]

        predictions = []
        for img_path in image_paths:
            features = self.extract_cnn_features(img_path)
            pred_scaled = self.model.predict(features.reshape(1, -1))[0]
            pred = self.scaler.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(pred)

        for pred, true in zip(predictions, true_tensions):
            print(f"Predicted Tension: {pred:.2f} -> True: {true:.2f}")

        return predictions

    def train(self):
        raise NotImplementedError("Prediction-only class. Use `XGBoostModel` to train.")

    def save(self, save_path: str):
        raise NotImplementedError("Model should be trained before saving.")