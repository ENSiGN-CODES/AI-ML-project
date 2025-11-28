import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# -------------------------------------------------
# 1️⃣ MODEL ARCHITECTURE (EXTRACTED FROM YOUR CODE)
# -------------------------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(6400, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x  # logits


# -------------------------------------------------
# 2️⃣ LOAD TRAINED MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = CNN(num_classes=3)
    model.load_state_dict(torch.load("cnn_brain_tumor.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]


# -------------------------------------------------
# 3️⃣ IMAGE PREPROCESSING (EXTRACTED FROM YOUR CODE)
# -------------------------------------------------
def preprocess_image(img_pil):
    img = np.array(img_pil.convert("L"))  # grayscale

    # Resize to training size
    img = cv2.resize(img, (512, 512))

    # Normalize
    max_val = img.max() if img.max() > 0 else 1
    img = img.astype(np.float32) / max_val

    # Add channel (1,512,512)
    img = np.expand_dims(img, axis=0)

    # Convert to tensor
    img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img_t


# -------------------------------------------------
# 4️⃣ STREAMLIT UI
# -------------------------------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload a brain MRI image to detect tumor type.")

uploaded = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded MRI Image", width=300)

    tensor = preprocess_image(img)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).numpy()[0]
        pred = np.argmax(probs)
        confidence = probs[pred]

    st.subheader("Prediction Result")
    st.success(f"Tumor Type: **{CLASS_NAMES[pred]}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")

st.markdown("---")
st.write("Developed by Aanya Jain — AIML Project")
