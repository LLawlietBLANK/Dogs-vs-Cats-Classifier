import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import os
import requests
from tensorflow.keras.models import load_model

MODEL_URL = "https://huggingface.co/your-username/your-repo-name/resolve/main/cat_dog_model.keras"
MODEL_PATH = "cat_dog_model.keras"

@st.cache_resource
def load_cat_dog_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
    return load_model(MODEL_PATH)

model = load_cat_dog_model()

st.title("🐱 Cat vs 🐶 Dog Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = img.resize((256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Classifying..."):
        prediction = model.predict(img_array)[0][0]
        label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
        st.markdown(f"### Prediction: {label} ({prediction:.2f})")
