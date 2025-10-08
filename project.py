import streamlit as st
import tensorflow as tf
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    from tensorflow import keras
    model = keras.models.load_model("cnn_model.h5", compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


model = load_model()

# Streamlit UI
st.title("🫁 Pneumonia Classification App")
st.write("Upload a Chest X-ray image to check if the patient has Pneumonia or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1️⃣ Load image as GRAYSCALE (same as training)
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    # 2️⃣ Preprocess the image
    img = img.resize((224, 224))
    img_array = img_to_array(img)  # shape (224, 224, 1)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 1)
    img_array = img_array / 255.0

    st.write("✅ Image shape:", img_array.shape)
    st.write("✅ Pixel range:", np.min(img_array), "-", np.max(img_array))

    # 3️⃣ Make prediction
    with st.spinner("Analyzing X-ray..."):
        prediction = model.predict(img_array)
        st.write("Raw model output:", prediction)

    st.success("Analysis complete!")

    # 4️⃣ Interpret results
    if prediction[0][0] >= 0.5:
        result = "Pneumonia"
        confidence = float(prediction[0][0])
    else:
        result = "Normal"
        confidence = 1 - float(prediction[0][0])

    # 5️⃣ Display results
    st.write(f"### 🩺 Prediction: {result}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Sidebar info
    st.sidebar.header("About This Model")
    st.sidebar.write("""
    - Model Type: CNN  
    - Input Size: 224×224 (Grayscale)  
    - Output: Pneumonia / Normal  
    - Framework: TensorFlow / Keras
    """)
