# ==============================================================================
# CliniScan: Lung-Abnormality Classification & Detection (with X-ray Validator)
# ==============================================================================
import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import gdown
import plotly.graph_objects as go
import numpy as np

# Lazy import for TensorFlow
def load_validator_model_lazy(path):
    from tensorflow.keras.models import load_model
    return load_model(path)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Classification model
CLASS_MODEL_PATH = "resnet50_multilabel_vinbigdata.pt"
CLASS_DRIVE_ID = "1CG3_0OJe5hc2JyXsyerHc0DZcSTfKt1v"
CLASS_MODEL_URL = f"https://drive.google.com/uc?id={CLASS_DRIVE_ID}"

# Detection model
DETECT_MODEL_PATH = "best.pt"
DETECT_DRIVE_ID = "1Tt7-qfGC8509TGZTMIT_cWIovgVyRiyc"
DETECT_MODEL_URL = f"https://drive.google.com/uc?id={DETECT_DRIVE_ID}"

# Validator model (.h5)
VALIDATOR_MODEL_PATH = "validator.h5"
VALIDATOR_DRIVE_ID = "19bfxZ6qirYPHA3wb09pzJh_4ePm92zgX"
VALIDATOR_MODEL_URL = f"https://drive.google.com/uc?id={VALIDATOR_DRIVE_ID}"

LOGO_PATH = "assets/logo.jpg"

# 14 lung abnormality classes
classes = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"
]

# ------------------------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CliniScan: Lung-Abnormality Analysis",
    layout="centered",
    initial_sidebar_state="auto"
)

# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    st.markdown("### Select Mode")
    mode = st.radio("Choose analysis type", ["Classification", "Detection"])
    st.markdown("---")
    st.markdown("### Confidence Threshold")
    confidence_threshold = st.slider(
        "Select minimum confidence", 0.0, 1.0, 0.5 if mode=="Classification" else 0.25, 0.05
    )
    st.markdown("---")
    st.markdown("#### About the Model")
    if mode=="Classification":
        st.info("ResNet50 multi-label classifier for lung abnormalities.")
    else:
        st.info("YOLOv8 object detection for 14 lung abnormalities.")
    st.markdown("---")
    st.markdown("#### Disclaimer")
    st.warning("For educational/research purposes only. Not a substitute for professional medical advice.")

# ------------------------------------------------------------------------------
# Download model weights if not present
# ------------------------------------------------------------------------------
for path, url, desc in [
    (CLASS_MODEL_PATH, CLASS_MODEL_URL, "classification model"),
    (DETECT_MODEL_PATH, DETECT_MODEL_URL, "detection model"),
    (VALIDATOR_MODEL_PATH, VALIDATOR_MODEL_URL, "validator model")
]:
    if not os.path.exists(path):
        with st.spinner(f"Downloading {desc}... Please wait"):
            gdown.download(url, path, quiet=False)

# ------------------------------------------------------------------------------
# Load models
# ------------------------------------------------------------------------------
@st.cache_resource
def load_classification_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(CLASS_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_detection_model():
    return YOLO(DETECT_MODEL_PATH)

@st.cache_resource
def load_validator_model():
    return load_validator_model_lazy(VALIDATOR_MODEL_PATH)

# Lazy load models
if mode == "Classification":
    model, device = load_classification_model()
else:
    model = load_detection_model()

validator_model = load_validator_model()

# ------------------------------------------------------------------------------
# Image transform for classification
# ------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------------------------------------------------------------------
# Main interface
# ------------------------------------------------------------------------------
st.markdown("## CliniScan: Lung-Abnormality Analysis")
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

def preprocess_validator(img):
    from tensorflow.keras.preprocessing.image import img_to_array
    img_resized = img.resize((224, 224))
    arr = img_to_array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- Step 1: Validate chest X-ray ---
    img_input = preprocess_validator(image)
    pred = validator_model.predict(img_input)[0][0]  # sigmoid output
    if pred < 0.5:
        st.error("⚠️ Uploaded image is NOT recognized as a Chest X-ray. Please upload a valid X-ray image.")
    else:
        st.success("✅ Image validated as Chest X-ray. Proceeding with analysis...")

        # --- Step 2: Classification ---
        if mode == "Classification":
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = torch.sigmoid(model(img_tensor)).cpu().numpy()[0]

            preds = [(cls_name, float(conf)) for cls_name, conf in zip(classes, outputs)
                     if conf >= confidence_threshold]

            if preds:
                st.subheader("Predicted Abnormalities")
                df = pd.DataFrame(sorted(preds, key=lambda x: x[1], reverse=True),
                                  columns=["Abnormality", "Confidence"])
                st.table(df)

                fig = go.Figure(go.Bar(
                    x=df["Confidence"],
                    y=df["Abnormality"],
                    orientation='h',
                    marker=dict(color=df["Confidence"], colorscale='YlGnBu'),
                    hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}<extra></extra>'
                ))
                fig.update_layout(title="Confidence Scores by Abnormality",
                                  xaxis_title="Confidence", yaxis_title="Abnormality",
                                  template="plotly_white", height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No abnormalities detected above the selected confidence threshold.")

        # --- Step 3: Detection ---
        else:
            results = model.predict(image, imgsz=640, conf=confidence_threshold)
            res = results[0]
            annotated = res.plot()
            st.image(annotated, caption="Detected Abnormalities", use_container_width=True)

            if len(res.boxes) > 0:
                st.subheader("Detected Findings")
                det_summary = {}
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if conf >= confidence_threshold:
                        cls_name = classes[cls_id]
                        det_summary[cls_name] = max(det_summary.get(cls_name, 0), conf)
                        st.write(f"- {cls_name} ({conf:.2f})")

                df = pd.DataFrame(sorted(det_summary.items(), key=lambda x: x[1], reverse=True),
                                  columns=["Abnormality", "Confidence"])
                fig = go.Figure(go.Bar(
                    x=df["Confidence"], y=df["Abnormality"], orientation='h',
                    marker=dict(color=df["Confidence"], colorscale='YlGnBu',
                                line=dict(color='rgba(0, 0, 0, 0.6)', width=1)),
                    hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}<extra></extra>'
                ))
                fig.update_layout(title="Confidence Scores by Abnormality",
                                  xaxis_title="Confidence", yaxis_title="Abnormality",
                                  template="plotly_white", height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No abnormalities detected above the selected confidence threshold.")
