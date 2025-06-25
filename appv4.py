import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import requests
from pathlib import Path

# Page settings
st.set_page_config(page_title="Plant Disease Classification Demo", layout="wide")
st.title("Plant Disease Classification Demo")

# ---------------- MODEL OPTIONS ----------------
# Both models are stored in the same HF repo

MODEL_OPTIONS = {
    "ResNet50": "resnet50_Color_Model.h5",
    "MobileNetV2": "mobilenetv2_Color_Model.h5",
    "MobileNetV2-Grayscale": "mobilenetv2_Grayscale_Model.h5"
}
BASE_URL = "https://huggingface.co/oferaskgil/Models/resolve/main"

# Sidebar: choose model
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Select model", list(MODEL_OPTIONS.keys()))
filename = MODEL_OPTIONS[model_choice]

MODEL_URL = f"{BASE_URL}/{filename}"
LABELS_URL = f"{BASE_URL}/labels.txt"

# Utility: download remote file to temp and cache it
@st.cache_resource
def download_file(url: str, suffix: str) -> str:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                temp.write(chunk)
    temp.close()
    return temp.name

# Download model & labels
with st.spinner(f"Downloading {model_choice} model..."):
    model_path = download_file(MODEL_URL, ".h5")
with st.spinner("Downloading labels file..."):
    labels_path = download_file(LABELS_URL, ".txt")

# Load model once and cache
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)

model = load_model(model_path)

# Load class names
def load_labels(path: str) -> list:
    raw = Path(path).read_text(encoding="utf-8")
    return [lbl.strip() for lbl in raw.split("|") if lbl.strip()]

class_names = load_labels(labels_path)

# Sidebar status
st.sidebar.success(f"Loaded {model_choice}")
st.sidebar.write(f"{len(class_names)} classes available")

# Validate class count
n_out = model.output_shape[-1]
if len(class_names) != n_out:
    st.error(f"Mismatch: labels ({len(class_names)}) vs model outputs ({n_out})")
    st.stop()

# Image upload
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Please upload an image to classify.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Uploaded Image", use_column_width=True)


# Preprocess
img_resized = img.resize((224, 224))
arr = np.array(img_resized) / 255.0
input_tensor = np.expand_dims(arr, axis=0)

# Predict
preds = model.predict(input_tensor)[0]
top_indices = preds.argsort()[-3:][::-1]

# Display top 3
st.subheader("Top 3 Predictions")
for idx in top_indices:
    st.write(f"{class_names[idx]}: {preds[idx]:.1%}")

# Display all probabilities
st.subheader("All Class Probabilities")
df = pd.DataFrame({'Probability': preds}, index=class_names)
st.bar_chart(df)

# Health check
best_label = class_names[top_indices[0]]
best_score = preds[top_indices[0]]
if "healthy" in best_label.lower():
    st.success(f"✅ Healthy – {best_label} ({best_score:.1%})")
else:
    st.error(f"❌ Sick – {best_label} ({best_score:.1%})")


