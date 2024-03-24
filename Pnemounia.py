import streamlit as st
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
import requests

# Load the model and feature extractor
model_name = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Function to make predictions
def predict(image):
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Make prediction
    outputs = model(**inputs)
    predicted_class_idx = torch.argmax(outputs.logits)

    # Return prediction
    return predicted_class_idx.item()

# Streamlit App
def main():
    st.title("Pneumonia Detection from Chest X-ray")
    st.write("Upload a chest X-ray image to detect pneumonia.")

    # Upload image
    uploaded_image = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction on the uploaded image
        if st.button('Detect Pneumonia'):
            prediction_idx = predict(image)
            if prediction_idx == 1:
                st.write("Pneumonia Detected")
            else:
                st.write("No Pneumonia Detected")

if __name__ == "__main__":
    main()
