import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Load your PyTorch model
model = torch.load('model.pth', map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

# Define your class names (adjust based on your model's classes)
class_names = ['Disease 1', 'Disease 2', 'Disease 3', 'Healthy']

def preprocess_image(image):
    # Define the same transforms used during training
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def predict_disease(image):
    with torch.no_grad():
        preprocessed_img = preprocess_image(image)
        outputs = model(preprocessed_img)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = torch.max(probabilities).item()
        predicted_class = class_names[predicted.item()]
    return predicted_class, confidence

def get_treatment(disease):
    # Define treatments for each disease
    treatments = {
        'Disease 1': 'Treatment for Disease 1...',
        'Disease 2': 'Treatment for Disease 2...',
        'Disease 3': 'Treatment for Disease 3...',
        'Healthy': 'No treatment needed. The crop appears healthy.'
    }
    return treatments.get(disease, 'Unknown disease. Please consult an expert.')

st.title('Crop Disease Prediction')

uploaded_file = st.file_uploader("Choose an image of the crop", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict Disease'):
        predicted_disease, confidence = predict_disease(image)
        st.write(f"Predicted Disease: {predicted_disease}")
        st.write(f"Confidence: {confidence:.2f}")
        
        treatment = get_treatment(predicted_disease)
        st.write("Recommended Action:")
        st.write(treatment)