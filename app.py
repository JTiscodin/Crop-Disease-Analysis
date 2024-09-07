import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Define your class names (adjust based on your model's classes)
class_names = ['Disease 1', 'Disease 2', 'Disease 3', 'Healthy']

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, len(class_names))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + self.res2(x)
        x = self.classifier(x)
        return x

# Load your PyTorch model
@st.cache_resource
def load_model():
    model = CustomModel()
    state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

def preprocess_image(image):
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