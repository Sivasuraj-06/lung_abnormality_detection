# ==============================================================================
# CliniScan: Lung-Abnormality Classification & Detection (ONNX Version)
# ==============================================================================
import os
import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import gdown
import plotly.graph_objects as go
from torchvision import transforms

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# ONNX Models
CLASS_MODEL_PATH = "resnet50_multilabel_vinbigdata.onnx"
CLASS_DRIVE_ID = "1CG3_0OJe5hc2JyXsyerHc0DZcSTfKt1v"
CLASS_MODEL_URL = f"https://drive.google.com/uc?id={CLASS_DRIVE_ID}"

DETECT_MODEL_PATH = "best.onnx"
DETECT_DRIVE_ID = "1Tt7-qfGC8509TGZTMIT_cWIovgVyRiyc"
DETECT_MODEL_URL = f"https://drive.google.com/uc?id={DETECT_DRIVE_ID}"

LOGO_PATH = "assets/logo.jpg"

# Define the 14 class names
classes = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"
]

# ------------------------------------------------------------------------------
# Page Setup
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
    st.image(LOGO_PATH, use_container_width=True)
    st.markdown("### Select Mode")
    mode = st.radio("Choose analysis type", ["Classification", "Detection"])
    st.markdown("---")
    confidence_threshold = st.slider(
        "Select minimum confidence", 0.0, 1.0, 0.5 if mode=="Classification" else 0.25, 0.05
    )
    st.markdown("---")
    st.markdown("#### About the Model")
    if mode=="Classification":
        st.info("ResNet50-based multi-label classifier for 14 lung abnormalities.")
    else:
        st.info("YOLOv8 detection model for 14 lung abnormalities.")
    st.markdown("---")
    st.markdown("#### Disclaimer")
    st.warning("For educational/research purposes only. Not a substitute for professional medical advice.")

# ------------------------------------------------------------------------------
# Download models if not present
# ------------------------------------------------------------------------------
if not os.path.exists(CLASS_MODEL_PATH):
    with st.spinner("Downloading classification model..."):
        gdown.download(CLASS_MODEL_URL, CLASS_MODEL_PATH, quiet=False)

if not os.path.exists(DETECT_MODEL_PATH):
    with st.spinner("Downloading detection model..."):
        gdown.download(DETECT_MODEL_URL, DETECT_MODEL_PATH, quiet=False)

# ------------------------------------------------------------------------------
# Load ONNX Models
# ------------------------------------------------------------------------------
@st.cache_resource
def load_classification_model():
    return ort.InferenceSession(CLASS_MODEL_PATH)

@st.cache_resource
def load_detection_model():
    return ort.InferenceSession(DETECT_MODEL_PATH)

if mode == "Classification":
    ort_class_model = load_classification_model()
else:
    ort_det_model = load_detection_model()

# ------------------------------------------------------------------------------
# Image Transform for Classification
# ------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------------------------------------------------------------------
# Main Interface
# ------------------------------------------------------------------------------
st.markdown("## CliniScan: Lung-Abnormality Analysis")
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if mode == "Classification":
        # Multi-label classification (ONNX)
        img_tensor = transform(image).unsqueeze(0).numpy()
        outputs = ort_class_model.run(None, {"input": img_tensor})[0][0]
        preds = [(cls_name, float(conf)) for cls_name, conf in zip(classes, outputs) if conf >= confidence_threshold]

        if preds:
            st.subheader("Predicted Abnormalities")
            preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
            df = pd.DataFrame(preds_sorted, columns=["Abnormality", "Confidence"])
            st.table(df)

            # Horizontal bar chart
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

    else:
        # YOLO Detection (ONNX)
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_input = img_np.astype(np.float32) / 255.0  # normalize
        img_input = np.transpose(img_input, (2, 0, 1))[None, :, :, :]  # CHW + batch
        outputs = ort_det_model.run(None, {"images": img_input})[0]

        # You can implement YOLO postprocessing (boxes, scores, class IDs) here
        # For simplicity, just showing placeholder
        st.subheader("Detection results")
        st.write("YOLO ONNX inference complete. You can implement postprocessing to extract boxes, scores, and labels.")

        # If you want, I can also provide a **full YOLOv8 ONNX postprocessing snippet** for Streamlit.

