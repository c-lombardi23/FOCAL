"""This module enables displaying a heatmap for an image."""

from cleave_app.grad_cam import gradcam_driver

from .base_command import BaseCommand


class GradCamDisplay(BaseCommand):
    """Generate GradCAM visualization."""

    def _execute_command(self, config) -> None:
        if config.img_path and config.test_features:
            gradcam_driver(
                config.model_path,
                config.img_path,
                config.test_features,
                backbone_name=config.backbone_name,
                class_index=0,
                conv_layer_name=None,
                heatmap_file="C:\\Users\\clombardi\\heatmap_7_1.png",
            )
        else:
            print("Missing image path or test features for GradCAM")
