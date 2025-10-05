# digit_predictor.py (only get_model changed)
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import tensorflow as tf

_MODEL = None
_MODEL_PATH = Path(__file__).with_name("digit_recognizer.h5")
_BUILT = False

def get_model():
    global _MODEL, _BUILT
    if _MODEL is None:
        _MODEL = load_model(_MODEL_PATH, compile=False)
    if not _BUILT:
        # one dummy forward pass to create concrete inputs/outputs
        dummy = np.zeros((1, 28, 28, 1), dtype=np.float32)
        _ = _MODEL(tf.convert_to_tensor(dummy), training=False)
        _BUILT = True
    return _MODEL

def preprocess_pil(pil_img, invert=True):
    """PIL Image -> (1,28,28,1) float32 in [0,1]."""
    img = pil_img.convert("L")
    if invert:  # canvas/upload often black on white
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    x = np.array(img).astype("float32") / 255.0
    x = x.reshape(1, 28, 28, 1)
    return x

def predict_topk_from_array(x, k=3):
    model = get_model()
    probs = model.predict(x, verbose=0)[0]
    idx = probs.argsort()[-k:][::-1]
    return [(int(i), float(probs[i])) for i in idx], probs
