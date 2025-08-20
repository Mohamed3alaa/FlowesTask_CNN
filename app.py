import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

# load your trained model
model = load_model("mymodel.keras")   # make sure the model file is in the same folder

# class labels (update according to your dataset)
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# streamlit app
st.title("🌸 Flower Classification App")
st.write("Upload a flower image and the model will classify it.")

# upload image
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocess
    img = image.resize((128, 128))   # resize to model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # result
    st.subheader("Prediction:")
    st.write(f"Flower type: **{class_names[predicted_class]}** 🌼")
    st.write(f"Confidence: **{confidence:.2f}**")
