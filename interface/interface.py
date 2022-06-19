import streamlit as st
from PIL import Image
import requests
import io
import time

def predict(image):
    print("Predicting...")
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
    return response_json, image_bytes

@st.cache(show_spinner=False)
def get_labels():
    print("Getting labels...")
    patience = 5
    labels = None
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
    return labels

labels = get_labels()

st.title("Grocery Classifier")
    
if labels is None:
    st.warning("Received error from server, labels could not be retrieved")
else:
    st.write("Labels:", labels)

image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if image_file is not None:
    st.image(image_file)

    st.subheader("Classification")
    
    if st.button("Predict"):
        st.session_state['response_json'], st.session_state['image_bytes'] = predict(image_file)

    if 'response_json' in st.session_state and st.session_state['response_json'] is not None:
        # Show the result
        st.markdown(f"**Prediction:** {st.session_state['response_json']['prediction']}")
        st.markdown(f"**Confidence:** {st.session_state['response_json']['confidence']}")
        
        # User feedback
        st.markdown("If this prediction was incorrect, please select below the correct label")
        correct_labels = labels.copy()
        correct_labels.remove(st.session_state['response_json']["prediction"])
        correct_label = st.selectbox("Correct label", correct_labels)
        if st.button("Submit"):
            # Save feedback
            response = requests.post(
                "http://localhost:5002/feedback",
                data={"correct_label": correct_label},
                files={"image": st.session_state['image_bytes']})
            if response.status_code == 200:
                st.success("Feedback submitted")
            else:
                st.error("Feedback could not be submitted. Error: {}".format(response.text))
                
        # Retrain from feedback
        if st.button("Retrain from feedback"):
            response = requests.post("http://localhost:5002/retrain")
            if response.status_code == 200:
                response_json = response.json()
                if response_json["status"] == "success":
                    st.success("Model retrained")
                else:
                    st.warning("Model could not be retrained. Error: {}".format(response_json["message"]))
            else:
                st.error("Model could not be retrained. Error: {}".format(response.text))