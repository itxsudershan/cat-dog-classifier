import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model('cat_vs_dog_model.keras')  # use your model name

# Title
st.title("🐱 Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog, and the model will try to predict it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    result = "🐶 Dog" if prediction[0][0] > 0.5 else "🐱 Cat"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

    # Output
    st.markdown(f"### Prediction: **{result}**")
    st.markdown(f"Confidence: `{confidence*100:.2f}%`")
    
