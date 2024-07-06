import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your pre-trained model here
# For example, replace 'your_model_path' with the path to your saved model
model = tf.keras.models.load_model('dog_cat_model.h5')

def predict_image(image):
    # Resize the image to the required input size
    input_img_resize = cv2.resize(image, (224, 224))
    input_img_scaled = input_img_resize / 255.0
    
    # Reshape the image for model prediction
    image_reshaped = np.reshape(input_img_scaled, (1, 224, 224, 3))
    
    # Predict the class of the image
    input_prediction = model.predict(image_reshaped)
    input_pred_label = np.argmax(input_prediction)
    
    # Return the prediction label
    return input_pred_label

def main():
    st.title("Dog vs Cat Classifier")
    
    # File uploader to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_img = cv2.imdecode(file_bytes, 1)
        
        # Display the uploaded image
        st.image(input_img, channels="BGR", caption="Uploaded Image", use_column_width=True)
        
        # Button for prediction
        if st.button("Predict"):
            input_pred_label = predict_image(input_img)
            
            # Display the result
            if input_pred_label == 0:
                st.write("The image contains a Dog")
            else:
                st.write("The image contains a Cat")

if __name__ == "__main__":
    main()
