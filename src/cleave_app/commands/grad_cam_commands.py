"""This module enables displaying a heatmap for an image."""

from cleave_app.grad_cam import GradCAM

from .base_command import BaseCommand


class GradCamDisplay(BaseCommand):
    """Generate GradCAM visualization."""

    def _execute_command(self, config) -> None:
        grad = GradCAM(
            class_index=config.class_index,
            model_path=config.model_path,
            backbone=config.backbone,
            conv_layer_name=config.conv_layer_name,
            image_folder=config.img_folder
        )
        if config.multiple_images == "y":
            grad.compute_all_heatmaps(
                save_path=config.save_path
        )
        else:
            grad.compute_heatmap(
                image_path=config.image_path,
                title=config.title,
                fig_size=config.fig_size
            )
   