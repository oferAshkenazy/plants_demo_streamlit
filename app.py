import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Title and Sidebar ---
st.title("Plant Disease Classification Demo")
# IMPORTANT: model_dir must point to the folder containing your .h5 model(s) and labels.txt,
# not to your dataset image directory. Mixing these will cause class-name mismatches.
model_dir = st.sidebar.text_input("Model directory", value="models")

# --- Load Model ---
@st.cache_resource
def load_model(path: str):
    model = tf.keras.models.load_model(path, compile=False)
    return model

# Find first .h5 model in folder
model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
if not model_files:
    st.error(f"No .h5 models found in '{model_dir}'")
    st.stop()
model_path = os.path.join(model_dir, model_files[0])
model = load_model(model_path)

# --- Build CLASS_NAMES ---
# Read class names from a comma-delimited file instead of labels.txt
class_file = os.path.join(model_dir, "labels.txt")
if os.path.isfile(class_file):
    # Load comma-separated class names
    with open(class_file, encoding='utf-8') as f:
        raw = f.read()
    class_names = [name.strip() for name in raw.split('|') if name.strip()]
    #st.error(len(sorted(class_names)))

    # Debug: show what was loaded
    st.write(f"Loaded {len(class_names)} class names from {class_file}")
    st.write(class_names[:5], "...", f"({len(class_names)} total)")
else:
    st.error(f"Class names file not found at '{class_file}'. Please provide class_names.txt in comma-delimited format.")
    st.stop()

# Validate that class_names matches model output
if len(class_names) != model.output_shape[-1]:
    st.error(
        f"Number of class labels ({len(class_names)}) does not match model output classes ({model.output_shape[-1]}). "
        "Please supply a proper class_names.txt or correct your model directory."
    )
    st.stop()

# --- File Uploader ---
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # --- Preprocess ---
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized) / 255.0
    input_tensor = np.expand_dims(arr, axis=0)

    # --- Inference ---
    preds = model.predict(input_tensor)[0]
    top_indices = preds.argsort()[-3:][::-1]
    top_labels = [class_names[i] for i in top_indices]
    top_scores = [float(preds[i]) for i in top_indices]

    st.subheader("Top 3 Predictions")
    for lbl, scr in zip(top_labels, top_scores):
        st.write(f"{lbl}: {scr:.1%}")

    # --- Probability Bar Chart ---
    st.subheader("All Class Probabilities")
    prob_df = pd.DataFrame({
        "Probability": preds
    }, index=class_names)
    st.bar_chart(prob_df)

    # --- Healthy vs Sick Logic ---
    bias_dist = np.ones(len(class_names)) / len(class_names)
    best_idx, best_conf = top_indices[0], preds[top_indices[0]]
    best_label = class_names[best_idx]

    # Check for bias-only
    if best_idx in np.argsort(bias_dist)[-3:][::-1] and \
       np.allclose(preds[top_indices], bias_dist[top_indices], atol=1e-4):
        st.warning("⚠️ Model bias-only predictions detected. Retrain or fix preprocessing.")
    elif "healthy" in best_label.lower():
        st.success(f"✅ Healthy – {best_label} ({best_conf:.1%})")
    else:
        st.error(f"⚠️ Sick – {best_label} ({best_conf:.1%})")