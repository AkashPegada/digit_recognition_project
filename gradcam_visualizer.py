# gradcam_visualizer.py
import numpy as np
import tensorflow as tf

def gradcam(model, x, class_index=None, last_conv_name=None, upsample_to=(28, 28)):
    """
    Compute Grad-CAM without relying on model.inputs/outputs (Keras 3 safe).
    model: Keras Sequential/Functional CNN loaded from .h5
    x: (1,H,W,1) float32 in [0,1]
    last_conv_name: name of the last Conv2D layer (recommended to pass explicitly)
    returns: (cam[H,W] in [0,1], int class_index)
    """
    # 1) Pick last conv layer (auto if not provided)
    if last_conv_name is None:
        conv_layers = [l for l in model.layers
                       if "conv" in l.name.lower() or "conv2d" in l.__class__.__name__.lower()]
        if not conv_layers:
            raise ValueError("No Conv2D layers found; specify last_conv_name")
        last_conv = conv_layers[-1]
    else:
        last_conv = model.get_layer(last_conv_name)

    # 2) Forward pass layer-by-layer and capture activations of last_conv
    with tf.GradientTape() as tape:
        out = tf.convert_to_tensor(x, dtype=tf.float32)
        conv_act = None
        for layer in model.layers:
            out = layer(out, training=False)
            if layer is last_conv:
                conv_act = out

        preds = out  # final model output
        if class_index is None:
            class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]

    # 3) Gradients wrt conv feature maps
    grads = tape.gradient(loss, conv_act)            # (1,h,w,c)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # (c,)
    cam = tf.reduce_sum(weights * conv_act, axis=-1)[0]  # (h,w)

    # 4) Normalize & upsample
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], upsample_to, method="bilinear")[..., 0]
    return cam.numpy(), int(class_index)
