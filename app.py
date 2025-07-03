import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import os
import requests
from tensorflow.keras.models import load_model

MODEL_URL = "https://huggingface.co/LLawlietBLANK/Dogs-vs-Cats-Classifier/resolve/main/dogsVcats.keras"
MODEL_PATH = "dogsVcats.keras"

@st.cache_resource
def load_cat_dog_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
    return load_model(MODEL_PATH)

model = load_cat_dog_model()

# st.title("Cat vs Dog Classifier Using CNN")

# uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
# if uploaded_file:
#     img = Image.open(uploaded_file).convert("RGB")
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     # Preprocessing
#     img = img.resize((256, 256))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Prediction
#     with st.spinner("Classifying..."):
#         prediction = model.predict(img_array)[0][0]
#         label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
#         st.markdown(f"### Prediction: {label} ({prediction:.2f})")

st.set_page_config(page_title="Cat vs Dog Classifier", layout="wide")

# Title
st.markdown(
    "<h1 style='text-align: center; color: #4A4A4A;'>🐾 Cat vs Dog Classifier Using CNN</h1>",
    unsafe_allow_html=True,
)
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.markdown("### 📷 Uploaded Image")
    st.image(img, width=400, use_column_width=False)

    # Preprocessing
    img = img.resize((256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🔍 Classifying... Please wait"):
        prediction = model.predict(img_array)[0][0]
        label = "🐶 Dog" if prediction > 0.5 else "🐱 Cat"
        confidence = prediction if prediction > 0.5 else 1 - prediction

    st.success("✅ Classification Complete!")
    st.markdown(
        f"<h3 style='text-align: center;'>Prediction: {label}</h3>"
        f"<p style='text-align: center;'>Confidence: {confidence:.2%}</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown("👈 Upload an image to get started.")