import streamlit as st
import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd # Make sure pandas is imported for st.bar_chart

# --- 1. CONFIGURATION ---
# IMPORTANT: These settings MUST match your training and inference notebooks
IMG_SIZE = 512
N_CLASSES = 14
MODEL_NAME = 'efficientnet-b0'
MODEL_WEIGHTS_PATH = 'final_vinbigdata_classifier.pth'
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Assuming you have the class mapping stored somewhere. 
# For this example, we'll hardcode it.
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

# --- 2. MODEL ARCHITECTURE ---
# This class needs to be defined in your app.py file so it can load the weights.
class VinBigDataClassifier(torch.nn.Module):
    def __init__(self, n_classes, model_name):
        super().__init__()
        # Using from_pretrained to ensure it matches the Kaggle notebook
        self.model = EfficientNet.from_pretrained(model_name)
        num_ftrs = self.model._fc.in_features
        self.model._fc = torch.nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        return self.model(x)

# --- 3. PREDICTION FUNCTION ---
# This function handles the entire prediction pipeline
@st.cache_resource
def load_model():
    """
    Loads the model from the .pth file, handling various checkpoint formats.
    """
    model = VinBigDataClassifier(n_classes=N_CLASSES, model_name=MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    try:
        # Load the checkpoint, which may contain more than just the state dict
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        
        # Check if the checkpoint is a PyTorch Lightning checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Create a new state dict to handle key mismatches
        new_state_dict = {}
        for k, v in state_dict.items():
            # If the key does not start with 'model.', add it.
            if not k.startswith('model.'):
                new_state_dict['model.' + k] = v
            else:
                new_state_dict[k] = v
        
        # Load the new state dict into the model
        model.load_state_dict(new_state_dict)
            
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_WEIGHTS_PATH}' not found. Please ensure it is in the same directory.")
        return None, device
    except RuntimeError as e:
        st.error(f"A RuntimeError occurred while loading the model weights. The file might be corrupted or the model architecture does not match the weights.")
        st.error(f"Details: {e}")
        return None, device
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, device
        
    model.eval()
    return model, device

@st.cache_data
def preprocess_image(image):
    """Applies the necessary transforms to the input image."""
    inference_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    return inference_transforms(image).unsqueeze(0)

def predict(model, device, image):
    """Performs inference and returns predicted classes and probabilities."""
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


# --- 4. STREAMLIT APP LAYOUT ---
st.set_page_config(
    page_title="Chest X-ray Classifier",
    page_icon="🩺",
    layout="wide"
)

st.title("Chest X-ray Abnormality Classifier")
st.markdown("Upload a chest X-ray image to get a prediction of potential findings.")

# Load the model only once
model, device = load_model()

if model is None:
    st.stop()

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose a PNG, JPG, or JPEG file", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded X-ray")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Prediction Results")
        # Perform prediction on the image
        results, probabilities = predict(model, device, image)
        
        st.success("Analysis Complete!")
        st.markdown("### Predicted Findings:")
        for res in results:
            st.write(f"- {res}")

        st.markdown("---")
        st.markdown("### All Findings (with Confidence Scores):")
        
        # Display a bar chart of all probabilities
        prob_df = pd.DataFrame({
            'Class': list(idx_to_class.values()),
            'Confidence': probabilities,
        }).sort_values('Confidence', ascending=False)
        
        st.bar_chart(prob_df, x='Class', y='Confidence')
