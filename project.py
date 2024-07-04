import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import tensorflow_hub as hub
from PIL import Image

# Load your pre-trained model with custom objects
with custom_object_scope({'KerasLayer': hub.KerasLayer}):
    model = load_model('dog_cat_model.h5')

# Streamlit UI
st.title("Dog vs Cat Image Classification")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_img = cv2.imdecode(file_bytes, 1)
    
    # Display the uploaded image
    st.image(input_img, channels="BGR", caption="Uploaded Image")
    
    # Resize the image to 224x224 pixels
    input_img_resize = cv2.resize(input_img, (224, 224))
    
    # Normalize the image
    input_img_scaled = input_img_resize / 255.0
    
    # Reshape the image to match the input shape of the model
    image_reshaped = np.reshape(input_img_scaled, [1, 224, 224, 3])
    
    # Predict the image class
    input_prediction = model.predict(image_reshaped)
    
    # Get the predicted label
    input_pred_label = np.argmax(input_prediction)
    
    # Display the prediction and the label
    st.write(f"Prediction: {input_prediction}")
    st.write(f"Predicted Label: {input_pred_label}")
    
    if input_pred_label == 0:
        st.write("The image contains a Dog")
    else:
        st.write("The image contains a Cat")
