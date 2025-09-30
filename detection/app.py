# ==============================================================================
# Streamlit Web Application (Deployment) - app.py
# ==============================================================================
import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import gdown  # <-- NEW for downloading model

# ------------------------------------------------------------------------------
# Download model weights from Google Drive if not already present
# ------------------------------------------------------------------------------
MODEL_PATH = "best.pt"
DRIVE_ID = "1Tt7-qfGC8509TGZTMIT_cWIovgVyRiyc"  # 👈 paste your Google Drive file ID
MODEL_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ------------------------------------------------------------------------------
# Function to load YOLO model (cached for performance)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

# Define the 14 target class names (must match training config)
class_names = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"
]

# --- Streamlit UI ---
st.title("🩻 Chest X-Ray Abnormality Detection (YOLO)")
st.write("Upload a Chest X-ray image and the model will detect abnormalities.")

# File uploader widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run inference
    results = model.predict(image, imgsz=640, conf=0.25)
    res = results[0]

    # Show image with bounding boxes
    annotated = res.plot()
    st.image(annotated, caption="Detections", use_column_width=True)

    # Detection summary
    if len(res.boxes) > 0:
        st.write("### Detected Findings")
        det_summary = {}
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = class_names[cls_id]
            st.write(f"- {cls_name} ({conf:.2f})")
            det_summary[cls_name] = max(det_summary.get(cls_name, 0), conf)

        st.write("### Confidence Scores")
        st.bar_chart(det_summary)
    else:
        st.success("✅ No abnormalities detected")
