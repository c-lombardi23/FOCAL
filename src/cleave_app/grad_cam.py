"""This module defines the logic for displaying heatmaps for an image
to view where the CNN model is focusing on."""

import os
import sys
from typing import List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from tensorflow.keras.applications.efficientnet import preprocess_input
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore


class GradCAM:
    def __init__(
        self,
        model_path: str,
        image_folder: str,
        class_index: Optional[int]=0,
        backbone: Optional[str]=None,
        conv_layer_name: Optional[str]=None,
    ) -> None:
        """Initialize class and load model.

        Args:
            model_path: Path to classifier model
            image_folder: Contains images to be used for heatmaps
            class_index: Class number (0, 1) for binary classification
            backbone: Name of the pre-trained model
            conv_layer_name: Last convolutional layer name

        Raises:
            ValueError: No convolutional layer found
        """

        self.model = tf.keras.models.load_model(model_path)
        self.conv_layer_name = conv_layer_name
        self.class_index = class_index
        self.image_folder = image_folder

        if backbone is not None:
            self.backbone_layer = self.model.get_layer(backbone)
            if self.conv_layer_name is None:
                # Find last Conv2D in backbone
                for layer in reversed(self.backbone_layer.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        self.conv_layer_name = layer.name
                        break
                if self.conv_layer_name is None:
                    raise ValueError("No Conv2D layer found in the backbone.")
            self.target_layer = self.backbone_layer.get_layer(
                self.conv_layer_name
            )
        else:
            if self.conv_layer_name is None:
                for layer in reversed(self.model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        self.conv_layer_name = layer.name
                        break
                if self.conv_layer_name is None:
                    raise ValueError("No Conv2D layer found in the model.")
            self.target_layer = self.model.get_layer(self.conv_layer_name)

    def plot_heatmap(
        self, 
        title: str, 
        gradcam: Any,
        img_array: np.ndarray, 
        fig_size: List[int]
    ) -> None:
        """Plotting logic for individual heatmap.
        Args:
            title: Title of indivdual plot
            gradcam: gradcam object
            img_array: array of pixels
            fig_size: size of individual plot
        """

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title(title)
        heatmap = cm.jet(gradcam)[:, :, :3]
        heatmap = np.uint8(heatmap * 255)
        img_array = np.uint8(np.clip(img_array, 0, 255))
        superimposed_img = np.uint8(0.8 * img_array + 0.5 * heatmap)
        ax.imshow(superimposed_img)
        ax.axis("off")
        plt.show()

    def compute_heatmap(
        self, image_path: str, title: str, fig_size: List[int]
    ) -> None:
        """
        Computes the heatmap for a given image and parameter vector.

        Args:
            image (str): The input image path.
            title: Title of the plot
            fig_size: size of each plotted figure
        """
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)

        replace2linear = ReplaceToLinear()
        score = CategoricalScore(self.class_index)

        gradcam = GradcamPlusPlus(
            self.model, model_modifier=replace2linear, clone=True
        )
        try:
            cam = gradcam(
                score, img_array, penultimate_layer=self.conv_layer_name
            )
        except ValueError:
            print(f"{self.conv_layer_name} not in model summary!")
            exit
        cam = np.squeeze(cam)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

        self.plot_heatmap(
            title=title, gradcam=cam, img_array=img_array, fig_size=fig_size
        )

    def compute_all_heatmaps(self, save_path: str) -> None:
        """Computes heatmaps for a number of images.

        Args:
            save_path: path to save the computed heatmap plot
        """
        replace2linear = ReplaceToLinear()
        score = CategoricalScore(self.class_index)
        gradcam = GradcamPlusPlus(
            self.backbone_layer, model_modifier=replace2linear, clone=False
        )

        image_files = [
            f
            for f in os.listdir(self.image_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        num_images = len(image_files)
        ncols = 4
        nrows = int(np.ceil(num_images / ncols))

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3)
        )

        for i, filename in enumerate(image_files):
            row, col = divmod(i, ncols)
            ax = axs[row, col] if nrows > 1 else axs[col]

            image_path = os.path.join(self.image_folder, filename)
            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(224, 224)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            processed_img = preprocess_input(np.copy(img_array))
            try:
                cam = gradcam(
                    score, processed_img, penultimate_layer=self.target_layer
                )
            except ValueError:
                print(f"{self.target_layer} not in model summary!")
                sys.exit()
            cam = np.squeeze(cam)
            cam = (cam - cam.min()) / (cam.max() - cam.min())

            heatmap = cm.jet(cam)[:, :, :3]
            heatmap = np.uint8(heatmap * 255)
            superimposed_img = np.uint8(0.6 * img_array + 0.4 * heatmap)

            ax.imshow(superimposed_img)
            ax.set_title(filename, fontsize=8)
            ax.axis("off")

        for j in range(i + 1, nrows * ncols):
            row, col = divmod(j, ncols)
            ax = axs[row, col] if nrows > 1 else axs[col]
            ax.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            print("Plots not saved")
        plt.show()
