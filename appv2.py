import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Title and Sidebar ---
st.title("Plant Disease Classification Demo")

# --- Sidebar: choose your models directory ---
DEFAULT_MODEL_DIR = r"C:\Users\oferg\Desktop\Final\plants_demo_streamlit\Models"
model_dir = st.sidebar.text_input(
    "Models directory (folder containing .h5 and labels.txt)",
    value=DEFAULT_MODEL_DIR
)

# Validate the directory exists
if not os.path.isdir(model_dir):
    st.sidebar.error(f"Directory not found: {model_dir}")
    st.stop()

# --- Load Model ---
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)

# Find first .h5 model in folder
model_files = [f for f in os.listdir(model_dir) if f.lower().endswith(".h5")]
if not model_files:
    st.error(f"No .h5 models found in '{model_dir}'")
    st.stop()

model_path = os.path.join(model_dir, model_files[0])
model = load_model(model_path)

# --- Build CLASS_NAMES ---
class_file = os.path.join(model_dir, "labels.txt")
if not os.path.isfile(class_file):
    st.error(f"labels.txt not found in '{model_dir}'. Please add a pipe-delimited labels.txt.")
    st.stop()

with open(class_file, encoding='utf-8') as f:
    raw = f.read()
class_names = [name.strip() for name in raw.split('|') if name.strip()]

st.sidebar.write(f"Using model: **{model_files[0]}**")
st.sidebar.write(f"{len(class_names)} classes loaded")

# Validate classes vs model output
n_classes = model.output_shape[-1]
if len(class_names) != n_classes:
    st.error(
        f"Mismatch: labels.txt contains {len(class_names)} names, "
        f"but model outputs {n_classes} classes."
    )
    st.stop()

# --- File Uploader ---
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Please upload an image to classify.")
    st.stop()

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

# --- Show Top 3 ---
st.subheader("Top 3 Predictions")
for lbl, scr in zip(top_labels, top_scores):
    st.write(f"{lbl}: {scr:.1%}")

# --- All Probabilities ---
st.subheader("All Class Probabilities")
prob_df = pd.DataFrame({"Probability": preds}, index=class_names)
st.bar_chart(prob_df)

# --- Healthy vs Sick Logic ---
bias_dist = np.ones(len(class_names)) / len(class_names)
best_idx, best_conf = top_indices[0], preds[top_indices[0]]
best_label = class_names[best_idx]

if np.allclose(preds[top_indices], bias_dist[top_indices], atol=1e-4):
    st.warning("⚠️ Model appears to be predicting the prior—check your preprocessing or retrain.")
elif "healthy" in best_label.lower():
    st.success(f"✅ Healthy – {best_label} ({best_conf:.1%})")
else:
    st.error(f"⚠️ Sick – {best_label} ({best_conf:.1%})")
