from cleave_app.grad_cam import (
    gradcam_driver
)
import traceback

class GradCamDisplay:    

    def execute(self, config) -> None:
        """Generate GradCAM visualization.

        Args:
            config: Configuration object containing GradCAM parameters
        """
        try:
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

        except Exception as e:
            print(f"Error during GradCAM generation: {e}")
            traceback.print_exc()
            raise
