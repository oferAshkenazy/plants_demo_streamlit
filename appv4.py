import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import requests
from pathlib import Path

import pandas as pd
import requests
from io import StringIO

def load_treatment_plan(
    url="https://huggingface.co/oferaskgil/Models/resolve/main/Leaf_Disease_Treatment_Plan.csv"
) -> pd.DataFrame:
    resp = requests.get(url); resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))

def normalize(s: str) -> str:
    # replace triple-underscores first, then single underscores
    return s.replace('___',' ').replace('_',' ').strip().lower()

def get_treatments(
    plant_name: str,
    disease_name: str,
    df: pd.DataFrame = None
) -> tuple[str|None, str|None]:
    if df is None:
        df = load_treatment_plan()

    # build normalized columns
    df = df.assign(
        plant_norm   = df['Plant'].astype(str).map(normalize),
        disease_norm = df['Disease'].astype(str).map(normalize),
    )

    p_norm = normalize(plant_name)
    d_norm = normalize(disease_name)

    match = df[(df['plant_norm'] == p_norm) & (df['disease_norm'] == d_norm)]
    if match.empty:
        return None, None

    row = match.iloc[0]
    return row['1st Generation Treatment'], row['2nd Generation Treatment']

df = load_treatment_plan()

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

# Open full-res image
img = Image.open(uploaded).convert("RGB")

# — Display at 50% size —
orig_w, orig_h = img.size
new_size = (orig_w // 2, orig_h // 2)
display_img = img.resize(new_size)

st.image(
    display_img,
    caption="Uploaded Image",
    use_container_width =False,  # let width param control sizing
    width=new_size[0]
)

# Preprocess at full-res (or your standard 224×224)
img_resized = img.resize((224, 224))
arr = np.array(img_resized) / 255.0
input_tensor = np.expand_dims(arr, axis=0)
# …continue with your prediction code…

# Predict
preds = model.predict(input_tensor)[0]
top_indices = preds.argsort()[-3:][::-1]

# Health check
best_label = class_names[top_indices[0]]
# 1. replace every underscore with a pipe
s_pipe = best_label.replace("___", "|")
# 2. split into two parts at the first pipe
first_part, second_part = s_pipe.split("|", 1)

best_score = preds[top_indices[0]]
status=""
if "healthy" in best_label.lower():
    status="Healthy"
#    st.subheader("Prediction : ", )
#    st.success(f"✅ Healthy – {best_label} Probability ({best_score:.1%})")
else:
    status="Sick"
#    st.subheader("Prediction : ", )
#    st.error(f"⚠️ Sick – {best_label} Probability ({best_score:.1%})")

#st.write(f"{best_label}: {best_score:.1%}")
if status=="Healthy":
    st.write(f"The leaf that you uploaded is <span style='color:blue;font-weight:bold'>{first_part} </span>,and it indicates that the plant is <span style='color:green;font-weight:bold'>{status} - ({best_score:.1%})</span>"  ,  unsafe_allow_html=True)
if status=="Sick":
    st.write(
        f"The leaf that you uploaded is <span style='color:blue;font-weight:bold'>{first_part} </span>,and it indicates that the plant is <span style='color:red;font-weight:bold'>{status}</span>",
        unsafe_allow_html=True)
    st.write(f"The name of the plant disease : <span style='color:red;font-weight:bold'>{second_part} - ({best_score:.1%})</span>",
    unsafe_allow_html=True)

st.write("")
# Fetch your treatments however you do it
first, second = get_treatments(first_part, second_part, df)

# — Row 1: the header in column 1, nothing in column 2 —
col1, col2 = st.columns([0.3, 0.7])
with col1:
    st.write("Recommended treatment methods:")
with col2:
    pass  # leave blank

# — Row 2: blank in col1 to “indent” into col2, then your two lines —
col1, col2 = st.columns([0.05, 0.95])
with col1:
    st.write("")  # spacer
with col2:
    st.write("1st Gen:", first)
    st.write("2nd Gen:", second)


# Display all probabilities
st.write("")
st.subheader("All Class Probabilities")
df = pd.DataFrame({'Probability': preds}, index=class_names)
st.bar_chart(df)


# Health check
#best_label = class_names[top_indices[0]]
#best_score = preds[top_indices[0]]
#if "healthy" in best_label.lower():
#    st.success(f"✅ Healthy – {best_label} ({best_score:.1%})")
#else:
#    st.error(f"❌ Sick – {best_label} ({best_score:.1%})")


