import streamlit as st
from PIL import Image

st.title("Grocery Classifier")

image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_file is not None:
    st.image(image_file)

    st.subheader("Classification")

    # Load using PIL
    image = Image.open(image_file)

    