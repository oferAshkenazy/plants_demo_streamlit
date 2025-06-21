import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import requests

st.title("Plant Disease Classification Demo (HuggingFace Loader)")

# -------------- CONFIG: HuggingFace direct URLs -------------
HF_REPO = "oferaskgil/resnet50_model"
HF_MODEL_FILENAME = "resnet50_model.h5"
HF_LABELS_FILENAME = "labels.txt"

MODEL_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_MODEL_FILENAME}"
LABELS_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_LABELS_FILENAME}"

def download_file(url, suffix):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                temp.write(chunk)
    temp.close()
    return temp.name

# ------------------ DOWNLOAD MODEL AND LABELS -----------------
with st.spinner("Downloading model (.h5) from HuggingFace..."):
    model_path = download_file(MODEL_URL, ".h5")

with st.spinner("Downloading labels.txt from HuggingFace..."):
    labels_path = download_file(LABELS_URL, ".txt")

# ------------------ LOAD MODEL AND LABELS ---------------------
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)

model = load_model(model_path)

with open(labels_path, encoding='utf-8') as f:
    raw = f.read()
class_names = [name.strip() for name in raw.split('|') if name.strip()]

st.sidebar.success("Model and labels loaded from HuggingFace.")
st.sidebar.write(f"{len(class_names)} classes loaded")

n_classes = model.output_shape[-1]
if len(class_names) != n_classes:
    st.error(
        f"labels.txt contains {len(class_names)} names, "
        f"but model outputs {n_classes} classes."
    )
    st.stop()

# ------------------ IMAGE UPLOAD AND PREDICTION -----------------
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.info("Please upload an image to classify.")
    st.stop()

img = Image.open(uploaded_image).convert("RGB")
st.image(img, caption="Uploaded Image", use_column_width=True)

img_resized = img.resize((224, 224))
arr = np.array(img_resized) / 255.0
input_tensor = np.expand_dims(arr, axis=0)

preds = model.predict(input_tensor)[0]
top_indices = preds.argsort()[-3:][::-1]
top_labels = [class_names[i] for i in top_indices]
top_scores = [float(preds[i]) for i in top_indices]

st.subheader("Top 3 Predictions")
for lbl, scr in zip(top_labels, top_scores):
    st.write(f"{lbl}: {scr:.1%}")

st.subheader("All Class Probabilities")
prob_df = pd.DataFrame({"Probability": preds}, index=class_names)
st.bar_chart(prob_df)

bias_dist = np.ones(len(class_names)) / len(class_names)
best_idx, best_conf = top_indices[0], preds[top_indices[0]]
best_label = class_names[best_idx]

if np.allclose(preds[top_indices], bias_dist[top_indices], atol=1e-4):
    st.warning("⚠️ Model appears to be predicting the prior—check your preprocessing or retrain.")
elif "healthy" in best_label.lower():
    st.success(f"✅ Healthy – {best_label} ({best_conf:.1%})")
else:
    st.error(f"⚠️ Sick – {best_label} ({best_conf:.1%})")
