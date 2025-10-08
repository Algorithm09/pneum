import streamlit as st
import tensorflow as tf
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# -------------------------------
# 🔹 Load the trained CNN model
# -------------------------------

def load_model():
    from tensorflow import keras
    model = keras.models.load_model("cnn_model.h5", compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

model = load_model()

# -------------------------------
# 🔹 Streamlit UI
# -------------------------------
st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="centered")

st.title("🫁 Pneumonia Classification App")
st.write("Upload a Chest X-ray image to check if the patient has **Pneumonia** or is **Normal**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1️⃣ Load image as GRAYSCALE (same as training)
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Uploaded Chest X-ray", use_container_width=True)

    # 2️⃣ Preprocess image (resize, ensure shape, normalize)
    img = img.resize((224, 224))
    img_array = img_to_array(img)

    # Ensure shape is (224, 224, 1)
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)

    
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    st.write("✅ Final image shape:", img_array.shape)
    st.write("✅ Pixel range:", f"{np.min(img_array):.3f} - {np.max(img_array):.3f}")

    # 3️⃣ Predict
    with st.spinner("🔍 Analyzing X-ray..."):
        prediction = model.predict(img_array)
        pred_value = float(prediction[0][0])

    # 4️⃣ Interpret results
    if pred_value >= 0.5:
        result = "Pneumonia"
        confidence = pred_value
    else:
        result = "Normal"
        confidence = 1 - pred_value

    # 5️⃣ Display results
    st.success(f"### 🩺 Prediction: **{result}**")
    st.progress(confidence)
    st.write(f"**Confidence:** {confidence:.2%}")

# -------------------------------
# 🔹 Sidebar Info
# -------------------------------
st.sidebar.header("🧠 Model Info")
st.sidebar.write("""
**Model Type:** CNN  
**Input Size:** 224×224 (Grayscale)  
**Output:** Pneumonia / Normal  
**Framework:** TensorFlow / Keras  
**Activation:** Sigmoid  
**Loss:** Binary Crossentropy
""")
