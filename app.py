import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

class_names = ['Tomato__Late_blight', 'Tomato_healthy', 'Grape_healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Soybean__healthy', 'Squash_Powdery_mildew', 'Potato_healthy', 'Corn(maize)Northern_Leaf_Blight', 'Tomato_Early_blight', 'Tomato_Septoria_leaf_spot', 'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry_Leaf_scorch', 'Peach_healthy', 'Apple_Apple_scab', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Bacterial_spot', 'Apple_Black_rot', 'Blueberry_healthy', 'Cherry(including_sour)Powdery_mildew', 'Peach_Bacterial_spot', 'Apple_Cedar_apple_rust', 'Tomato_Target_Spot', 'Pepper,_bell_healthy', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Potato__Late_blight', 'Tomato_Tomato_mosaic_virus', 'Strawberry_healthy', 'Apple_healthy', 'Grape_Black_rot', 'Potato_Early_blight', 'Cherry(including_sour)healthy', 'Corn(maize)Common_rust', 'Grape__Esca(Black_Measles)', 'Raspberry__healthy', 'Tomato_Leaf_Mold', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Pepper,_bell_Bacterial_spot', 'Corn(maize)_healthy']


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
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128)
            )
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
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512)
            )
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 38)
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

@st.cache_resource
def load_model():
    model = CustomModel()
    state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
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
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]
        
        # Debug: Print all class probabilities
        for i, prob in enumerate(probabilities[0]):
            st.write(f"{class_names[i]}: {prob.item():.4f}")
        
    return predicted_class, confidence.item()

st.title('Crop Disease Prediction')

uploaded_file = st.file_uploader("Choose an image of the crop", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict Disease'):
        predicted_disease, confidence = predict_disease(image)
        st.write(f"Predicted Disease: {predicted_disease}")
        st.write(f"Confidence: {confidence:.2f}")

        # Debug: Print model output
        st.write("Model Output:")
        with torch.no_grad():
            outputs = model(preprocess_image(image))
            st.write(outputs)