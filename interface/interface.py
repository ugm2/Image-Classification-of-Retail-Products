import streamlit as st
from PIL import Image
import requests
import io
import time

st.title("Grocery Classifier")
patience = 5
with st.spinner("Retrieving labels"):
    while True:
        try:
            response = requests.post("http://localhost:5002/get_labels")
            labels = set(response.json()["labels"])
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(2)
            patience -= 1
            if patience == 0:
                raise Exception("Could not connect to server. Make sure the server is running.")
            continue
        except:
            labels = None
            break
    
if labels is None:
    st.warning("Received error from server, labels could not be retrieved")
else:
    st.write("Labels:", labels)

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
    st.markdown(f"**Prediction:** {response_json['prediction']}")
    st.markdown(f"**Confidence:** {response_json['confidence']}")
    
    # User feedback
    st.markdown("If this prediction was incorrect, please select below the correct label")
    correct_labels = labels.copy()
    correct_labels.remove(response_json["prediction"])
    correct_label = st.selectbox("Correct label", correct_labels)
    if st.button("Submit"):
        # Save feedback
        response = requests.post("http://localhost:5002/feedback", json={"correct_label": correct_label}, files={"image": image_bytes})
        if response.status_code == 200:
            st.success("Feedback submitted")
        else:
            st.error("Feedback could not be submitted. Error: {}".format(response.text))