Chest X-Ray Abnormality Classification (EfficientNet)

This project demonstrates a complete end-to-end pipeline for multi-label classification on chest X-ray images. Using the VinBigData Chest X-ray dataset, an EfficientNet-B0 model is trained to classify the presence of 14 different lung abnormalities. The trained model is deployed via a Streamlit web application with Grad-CAM explainability for interpretability.

🚀 Project Overview

The goal is to accurately classify multiple thoracic abnormalities from a single chest X-ray image. The workflow includes:

Multi-label target engineering

Robust training with PyTorch Lightning

Deployment via Streamlit

Grad-CAM visualizations for model interpretability

Feature Details Model EfficientNet-B0 (Pre-trained) Dataset VinBigData Chest X-ray Task Multi-Label Image Classification Training Env Kaggle/Colab (PyTorch Lightning) Deployment Streamlit (Local Host) Explainability Grad-CAM Image Size 512x512 pixels 🛠️ Technologies & Dependencies

Python: 3.8+

Deep Learning: torch, pytorch-lightning

Model Architecture: efficientnet-pytorch

Explainability: grad-cam

Data Science: numpy, pandas, sklearn, tqdm

Image Handling: Pillow (PIL), torchvision

Deployment: streamlit

Installation

Clone the repository
git clone cd

Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn pip install torch torchvision pytorch-lightning efficientnet-pytorch grad-cam streamlit

Or using requirements.txt
pip install -r requirements.txt
⚙️ Usage

The project consists of two main phases:

Phase 1: Data Preparation & Training (Kaggle/Colab)

Convert object detection metadata into multi-label classification targets

Define PyTorch Dataset and DataLoaders with augmentations

Train EfficientNet-B0 using PyTorch Lightning

Save the optimized weights file (finalvinbigdataclassifier.pth)

Generate Grad-CAM visualizations for interpretability

Notebook steps:

Step Description 1-2 Setup environment, global config, seed 3 Multi-label target creation and Train/Val split 4-5 Dataset & DataLoader definition with augmentations 6 Define PyTorch Lightning model with EfficientNet-B0 7 Training with ModelCheckpoint, EarlyStopping 8 Save clean .pth weights (finalvinbigdataclassifier.pth) 9 Generate Grad-CAM visualizations Phase 2: Deployment (Local Machine)

Download Model Weights: finalvinbigdataclassifier.pth

The model weights (best.pt) are hosted on Google Drive and will be automatically downloaded by the Streamlit app.

Save Deployment Script: app.py (Streamlit code including model class)

Run Streamlit App:

streamlit run app.py

Open your browser at http://localhost:8501

⚡ Features

Predicts 14 lung abnormalities from a single X-ray

Multi-label predictions with confidence scores

Grad-CAM visualizations highlight important image regions

Simple Streamlit UI for image upload and inference

Link for trained weights: https://drive.google.com/file/d/1Mr4ojGw6djSPVBrPVFyiSDb0o9HHOji-/view?usp=drive_link

Snapshots of the application:

![Chest X-ray Classifier_page-0001](https://github.com/user-attachments/assets/42f24a72-6ded-406f-86ba-0e48ba788335)
