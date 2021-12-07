import streamlit as st
from PIL import Image
import requests
import io

st.title("Grocery Classifier")

image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_file is not None:
    st.image(image_file)

    st.subheader("Classification")

    # Load using PIL
    image = Image.open(image_file)

    # Image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    # Send the image to the model
    response = requests.post("http://localhost:5002/predict", files={"image": image_bytes})

    # Get the response
    response_json = response.json()

    # Show the result
    st.write(response_json)