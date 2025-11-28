import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# -------------------------------
# Model Architecture (MUST MATCH TRAINING)
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # logits


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=3)
    model.load_state_dict(torch.load("cnn_brain_tumor.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]

# -------------------------------
# Preprocess Function
# -------------------------------
def preprocess_image(img_pil):
    """ Convert PIL image → model input tensor """
    img = np.array(img_pil.convert("L"))  # grayscale
    
    # Resize to match training
    img = cv2.resize(img, (512, 512))

    # Normalize
    maxv = img.max() if img.max() > 0 else 1
    img = img.astype(np.float32) / maxv

    # Add channel dim → (1, 512, 512)
    img = np.expand_dims(img, axis=0)

    # Convert to torch tensor
    img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,1,512,512)
    return img_t


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification from MRI")
st.write("Upload an MRI scan below to classify the tumor type.")

uploaded = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded MRI Image", width=300)

    tensor = preprocess_image(img)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).numpy()[0]
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]

    st.subheader("Prediction Result:")
    st.success(f"Tumor Type: **{CLASS_NAMES[pred_class]}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")

st.markdown("---")
st.write("Developed by Aanya Jain — AIML Project")
