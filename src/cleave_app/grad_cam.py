import tensorflow as tf
import numpy as np
import cv2


class GradCAM:
    def __init__(
        self,
        model_path,
        class_index=0,
        backbone_name=None,
        conv_layer_name=None,
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.class_index = class_index
        self.backbone_name = backbone_name

        if backbone_name is not None:
            backbone = self.model.get_layer(backbone_name)
            if conv_layer_name is None:
                # Find last Conv2D in backbone
                for layer in reversed(backbone.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        conv_layer_name = layer.name
                        break
                if conv_layer_name is None:
                    raise ValueError("No Conv2D layer found in the backbone.")
            self.target_layer = backbone.get_layer(conv_layer_name)
        else:
            if conv_layer_name is None:
                for layer in reversed(self.model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        conv_layer_name = layer.name
                        break
                if conv_layer_name is None:
                    raise ValueError("No Conv2D layer found in the model.")
            self.target_layer = self.model.get_layer(conv_layer_name)

    def compute_heatmap(self, image, param_vector, eps=1e-8):
        """
        Computes the heatmap for a given image and parameter vector.

        Args:
            image (numpy.ndarray): The input image.
            param_vector (numpy.ndarray): The parameter vector.
            eps (float): A small constant to prevent division by zero.

        Returns:
            numpy.ndarray: The computed heatmap.
        """

        img = np.array(image, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img_resized = cv2.resize(
            img,
            (
                self.model.input[0].shape[1],
                self.model.input[0].shape[2],
            ),
        )

        input_image = np.expand_dims(img_resized, axis=0)

        backbone = self.model.get_layer(self.backbone_name)
        last_conv_layer = backbone.get_layer(self.target_layer.name)

        grad_model = tf.keras.models.Model(
            inputs=backbone.input, outputs=last_conv_layer.output
        )

        with tf.GradientTape() as tape:
            inputs = tf.convert_to_tensor(input_image, dtype=tf.float32)
            tape.watch(inputs)
            conv_outputs = grad_model(inputs)
            pooled_outputs = tf.reduce_mean(conv_outputs, axis=[1, 2])
            loss = tf.reduce_sum(pooled_outputs)

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = (heatmap - np.min(heatmap)) / (
            np.max(heatmap) - np.min(heatmap) + eps
        )
        heatmap = np.uint8(255 * heatmap)
        return heatmap

    def overlay_heatmap(
        self, heatmap, image, alpha=0.3, colormap=cv2.COLORMAP_JET
    ):
        """
        Overlays the heatmap on the image.

        Args:
            heatmap (numpy.ndarray): The heatmap to overlay.
            image (numpy.ndarray): The input image.
            alpha (float): The transparency of the heatmap.
            colormap (int): The colormap to use for the heatmap.

        Returns:
            numpy.ndarray: The overlaid image.
        """
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, colormap)
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
        return overlay


def gradcam_driver(
    model_path,
    image_path,
    param_vector,
    class_index,
    backbone_name=None,
    conv_layer_name=None,
    heatmap_file=None,
):
    """
    Driver function to compute and display the GradCAM overlay.

    Args:
        model_path (str): The path to the model.
        image_path (str): The path to the image.
        param_vector (numpy.ndarray): The parameter vector.
        class_index (int): The index of the class to compute the heatmap for.
        backbone_name (str): The name of the backbone layer.
        conv_layer_name (str): The name of the convolutional layer.
        heatmap_file (str): The path to save the heatmap.

    Raises:
        FileNotFoundError: If the image is not found.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    param_vector = np.array(param_vector, dtype=np.float32)

    gradcam = GradCAM(
        model_path=model_path,
        class_index=class_index,
        backbone_name=backbone_name,
        conv_layer_name=conv_layer_name,
    )

    heatmap = gradcam.compute_heatmap(img, param_vector)

    overlay = gradcam.overlay_heatmap(heatmap, img)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imshow("GradCAM Overlay", overlay_bgr)
    cv2.imwrite(heatmap_file, overlay_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compute_saliency_map(model, image_path, param_vector, class_index=3):
    """
    Computes the saliency map for a given image and parameter vector.

    Args:
        model (tensorflow.keras.Model): The model to compute the saliency map for.
        image_path (str): The path to the image.
        param_vector (numpy.ndarray): The parameter vector.
        class_index (int): The index of the class to compute the saliency map for.

    Returns:
        numpy.ndarray: The computed saliency map.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    if image.max() > 1.0:
        image = image / 255.0
    image = image.astype(np.float32)
    input_image = tf.convert_to_tensor(np.expand_dims(image, axis=0))
    param_vector = np.expand_dims(param_vector, axis=0).astype(np.float32)
    param_tensor = tf.convert_to_tensor(param_vector)

    input_image = tf.Variable(input_image)

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        preds = model([input_image, param_tensor])
        loss = preds[:, class_index]

    grads = tape.gradient(loss, input_image)[0]
    print(grads)

    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()

    saliency -= saliency.min()
    saliency /= saliency.max() + 1e-8
    saliency = (saliency * 255).astype(np.uint8)

    return saliency
