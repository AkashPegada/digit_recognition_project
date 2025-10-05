# streamlit_app.py
import os, io
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from digit_predictor import preprocess_pil, predict_topk_from_array, get_model
from gradcam_visualizer import gradcam

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🧠 Handwritten Digit Recognition")

# Sidebar mode
mode = st.sidebar.radio("Input mode", ["Upload image", "Draw on canvas"])
show_cam = st.sidebar.checkbox("Show Grad-CAM heatmap", value=True)
invert_input = st.sidebar.checkbox("Invert input (white bg, black digit)", value=True)
last_conv_name = st.sidebar.text_input("Last conv layer (optional)", "")

# Optional: show model layers to help pick correct last conv layer
from digit_predictor import get_model

if st.sidebar.checkbox("Show model layers", value=False):
    m = get_model()
    info = [(l.name, l.__class__.__name__) for l in m.layers]
    st.sidebar.write(info)

# ---- Input acquisition ----
pil_img = None
if mode == "Upload image":
    up = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if up:
        pil_img = Image.open(io.BytesIO(up.read()))
        st.image(pil_img, caption="Uploaded", width=200)

else:
    # drawing canvas
    from streamlit_drawable_canvas import st_canvas
    st.write("Draw a digit below (best with a mouse).")
    canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas.image_data is not None:
        # canvas returns RGBA np array
        pil_img = Image.fromarray(canvas.image_data.astype("uint8")).convert("RGB")
        st.image(pil_img, caption="Canvas snapshot", width=200)

# ---- Predict & visualize ----
if pil_img:
    x = preprocess_pil(pil_img, invert=invert_input)
    top3, probs = predict_topk_from_array(x, k=3)

    st.subheader("Top-3 predictions")
    st.write(" · ".join([f"**{d}** ({p:.3f})" for d, p in top3]))

    # Confidence bar chart
    fig, ax = plt.subplots()
    ax.bar(range(10), probs)
    ax.set_xticks(range(10))
    ax.set_ylim([0, 1])
    ax.set_title("Confidence Scores")
    st.pyplot(fig)

    # Grad-CAM
    if show_cam:
        model = get_model()
        cam, cls = gradcam(
            model, x,
            class_index=top3[0][0],
            last_conv_name=last_conv_name or None,
            upsample_to=(28, 28)
        )
        base = (x[0, ..., 0] * 255).astype(np.uint8)
        base_rgb = np.dstack([base] * 3)
        heat = (plt.cm.jet(cam)[..., :3] * 255).astype(np.uint8)
        alpha = 0.45
        overlay = (alpha * heat + (1 - alpha) * base_rgb).astype(np.uint8)
        st.image(overlay, caption=f"Grad-CAM (class {cls})", use_container_width=False, width=200)

st.divider()
if st.button("Shutdown App"):
    st.warning("Shutting down Streamlit…")
    os._exit(0)
