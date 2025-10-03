# ==============================================================================
# Streamlit Web Application (Deployment) - Chest X-ray Classifier (EfficientNet)
# ==============================================================================
import streamlit as st
import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
import gdown  # <-- NEW for downloading model from Google Drive

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
IMG_SIZE = 512
N_CLASSES = 14
MODEL_NAME = 'efficientnet-b0'
MODEL_WEIGHTS_PATH = 'best_classification.pth'
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Mapping from class index to human-readable class names
idx_to_class = {
    0: 'Aortic enlargement', 
    1: 'Cardiomegaly', 
    2: 'Pulmonary fibrosis', 
    3: 'Pneumothorax',
    4: 'Pleural thickening', 
    5: 'Pleural effusion', 
    6: 'No finding', 
    7: 'Nodule/Mass', 
    8: 'Infiltration',
    9: 'ILD', 
    10: 'Other lesion', 
    11: 'Atelectasis', 
    12: 'Emphysema', 
    13: 'Calcification'
}

# ------------------------------------------------------------------------------
# 2. MODEL ARCHITECTURE
# ------------------------------------------------------------------------------
class VinBigDataClassifier(torch.nn.Module):
    """
    EfficientNet-based classifier for chest X-ray abnormalities.
    """
    def __init__(self, n_classes, model_name):
        super().__init__()
        self.model = EfficientNet.from_pretrained(model_name)
        num_ftrs = self.model._fc.in_features
        self.model._fc = torch.nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        return self.model(x)

# ------------------------------------------------------------------------------
# 3. DOWNLOAD MODEL FROM GOOGLE DRIVE
# ------------------------------------------------------------------------------
DRIVE_ID = '1Mr4ojGw6djSPVBrPVFyiSDb0o9HHOji-'  # 👈 Google Drive file ID
MODEL_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"

with st.spinner("Downloading model from Google Drive... ⏳"):
    gdown.download(MODEL_URL, MODEL_WEIGHTS_PATH, quiet=False)

# ------------------------------------------------------------------------------
# 4. LOAD MODEL FUNCTION
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Loads the EfficientNet model with pretrained weights.
    Handles possible key mismatches in state_dict.
    """
    model = VinBigDataClassifier(n_classes=N_CLASSES, model_name=MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    try:
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Handle key prefix issues
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('model.'):
                new_state_dict['model.' + k] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device
    
    model.eval()
    return model, device

# ------------------------------------------------------------------------------
# 5. IMAGE PREPROCESSING
# ------------------------------------------------------------------------------
@st.cache_data
def preprocess_image(image):
    """
    Applies necessary transforms to the input image.
    """
    inference_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    return inference_transforms(image).unsqueeze(0)

# ------------------------------------------------------------------------------
# 6. PREDICTION FUNCTION
# ------------------------------------------------------------------------------
def predict(model, device, image):
    """
    Performs inference and returns predicted classes and probabilities.
    """
    img_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    predicted_indices = np.where(probabilities > 0.5)[0]
    results = []

    if len(predicted_indices) == 0:
        max_idx = np.argmax(probabilities)
        results.append(f"{idx_to_class[max_idx]} (Fallback: {probabilities[max_idx]:.3f})")
    else:
        for i in predicted_indices:
            results.append(f"{idx_to_class[i]} ({probabilities[i]:.3f})")
    
    return results, probabilities

# ------------------------------------------------------------------------------
# 7. STREAMLIT WEB APP LAYOUT
# ------------------------------------------------------------------------------
st.title("CliniScan:Lung-Abnormality Classification on Chest X‐rays using EfficientNet Model")
st.markdown("Upload a chest X-ray image to get a prediction of potential findings.")

# Load model
model, device = load_model()
if model is None:
    st.stop()

# File uploader widget
uploaded_file = st.file_uploader("Choose a PNG, JPG, or JPEG file", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded X-ray")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Prediction Results")
        results, probabilities = predict(model, device, image)
        st.success("Analysis Complete!")

        st.markdown("### Predicted Findings:")
        for res in results:
            st.write(f"- {res}")

        st.markdown("---")
        st.markdown("### All Findings (with Confidence Scores):")
        prob_df = pd.DataFrame({
            'Class': list(idx_to_class.values()),
            'Confidence': probabilities,
        }).sort_values('Confidence', ascending=False)
        st.bar_chart(prob_df, x='Class', y='Confidence')
