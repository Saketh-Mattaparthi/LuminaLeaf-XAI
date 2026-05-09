import numpy as np
import tensorflow as tf
import cv2
import os
import shutil
from config import Config

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output neuron with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Average gradients spatially
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, output_path, alpha=0.4):
    # Load the original image
    img = cv2.imread(img_path)
    
    # Resize heatmap to match original image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap * alpha + img
    
    # Save the superimposed image
    cv2.imwrite(output_path, superimposed_img)

def generate_gradcam(img_path, filename):
    """
    Wrapper function. If model is available, generates gradcam.
    If mock mode, simply creates a copy of the original image with a mock green tint overlay as a placeholder.
    """
    from predict import model, preprocess_image
    
    output_path = os.path.join(Config.GRADCAM_FOLDER, filename)
    
    if model is None:
        # Mock mode placeholder - simply copy original or add slight effect
        try:
            img = cv2.imread(img_path)
            # Create a mock heatmap (just a green tint for visualization purposes)
            overlay = img.copy()
            overlay[:] = (0, 255, 0) # Green BGR
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.imwrite(output_path, img)
        except Exception:
            shutil.copy(img_path, output_path)
        return filename
        
    try:
        img_array = preprocess_image(img_path)
        
        # Determine the last conv layer for InceptionV3
        # Usually it's 'mixed10'
        last_conv_layer_name = 'mixed10'
        
        # Verify the layer exists in the model
        layer_names = [layer.name for layer in model.layers]
        if last_conv_layer_name not in layer_names:
            # Fallback to the last layer before pooling if 'mixed10' is buried
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4: # 4D tensor (batch, height, width, channels)
                    last_conv_layer_name = layer.name
                    break
        
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        save_gradcam(img_path, heatmap, output_path)
        
        return filename
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        # On failure, just copy the original file to gradcam folder
        shutil.copy(img_path, output_path)
        return filename
