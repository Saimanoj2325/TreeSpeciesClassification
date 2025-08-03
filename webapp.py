import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# === CLASS NAMES ===
class_names = [
    "amla", "asopalav", "babul", "bamboo", "banyan", "bili", "cactus", "champa", "coconut", "garmalo",
    "gulmohor", "gunda", "jamun", "kanchan", "kesudo", "khajur", "mango", "motichanoti", "neem", "nilgiri",
    "other", "pilikaren", "pipal", "saptaparni", "shirish", "simlo", "sitafal", "sonmahor", "sugarcane", "vad"
]


# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('F:\\TSCwebapp\\resnet50v2_finetuned_30class_v2.keras')

model = load_model()

# === IMAGE PREPROCESSING ===
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha channel if exists
    return np.expand_dims(img_array, axis=0)  # Add batch dim

# === UI CONFIG ===
st.set_page_config(page_title="🌳 Tree Classifier", layout="centered")

st.markdown("<h1 style='text-align: center; color: green;'>🌿 Tree Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("Upload an image of a tree to classify it among 30 species.", unsafe_allow_html=True)

# === UPLOAD ===
uploaded_file = st.file_uploader("📷 Upload Tree Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    with st.spinner("🔍 Analyzing..."):
        processed = preprocess_image(img)
        preds = model.predict(processed)[0]
        pred_idx = np.argmax(preds)
        confidence = preds[pred_idx] * 100
        pred_class = class_names[pred_idx]

    # === LAYOUT: SIDE BY SIDE ===
    st.markdown("---")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown(f"<h3>🧠 Predicted Class:</h3><p style='font-size:22px; color:green;'><b>{pred_class}</b></p>", unsafe_allow_html=True)
        st.markdown(f"<h4>🔎 Confidence:</h4><p style='font-size:18px'>{confidence:.2f}%</p>", unsafe_allow_html=True)
