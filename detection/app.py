# ==============================================================================
# Streamlit Web Application (Deployment) - app.py
# ==============================================================================
import streamlit as st # Core library for building web applications
from ultralytics import YOLO # Model library
from PIL import Image # For handling image uploads and manipulation

# Function to load and cache the YOLO model (for performance)
@st.cache_resource
def load_model():
    # Load the best-performing weights file
    model = YOLO("best.pt") 
    return model

model = load_model()

# Define the 14 target class names (must match the training config)
class_names = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"
]

# --- Streamlit UI ---
st.title("🩻 Chest X-Ray Abnormality Detection (YOLO)")
st.write("Upload a Chest X-ray image...")

# File uploader widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run inference on the uploaded image
    results = model.predict(image, imgsz=640, conf=0.25)
    res = results[0]

    # Show image with drawn bounding boxes
    annotated = res.plot()
    st.image(annotated, caption="Detections", use_column_width=True)

    # Show detection details and confidence chart
    if len(res.boxes) > 0:
        st.write("### Detected Findings")
        det_summary = {}
        for box in res.boxes:
            # Extract class ID and confidence score
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = class_names[cls_id]
            st.write(f"- {cls_name} ({conf:.2f})")
            det_summary[cls_name] = max(det_summary.get(cls_name, 0), conf)
        st.write("### Confidence Scores")
        st.bar_chart(det_summary)
    else:
        st.success("✅ No abnormalities detected")