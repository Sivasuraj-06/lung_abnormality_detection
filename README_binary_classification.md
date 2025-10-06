CliniScan: Lung-Abnormality Classification on Chest X-rays (ResNet-18)

This project demonstrates a complete end-to-end pipeline for binary classification on chest X-ray images.
Using the VinBigData Chest X-ray dataset, a ResNet-18 model is trained to classify the presence of lung abnormalities (Abnormal vs. Normal).
The trained model is deployed via a Streamlit web application with Grad-CAM explainability for interpretability.

Project Overview

The goal is to accurately classify a single chest X-ray as either Normal or Abnormal.
The workflow includes:

Binary dataset preparation and balancing

Training using PyTorch (ResNet-18 backbone)

Deployment via Streamlit

Grad-CAM visualizations for clinical interpretability

Feature Details
Feature	Description
Model	ResNet-18 (Pre-trained on ImageNet)
Dataset	VinBigData Chest X-ray
Task	Binary Image Classification (Normal / Abnormal)
Training Env	Kaggle / Colab (PyTorch)
Deployment	Streamlit (Local Host)
Explainability	Grad-CAM
Image Size	224×224 pixels
Technologies & Dependencies

Python: 3.8+

Deep Learning: torch, torchvision

Data Augmentation: albumentations, opencv-python

Metrics: scikit-learn

Image Handling: Pillow (PIL)

Deployment: streamlit

Installation

Clone the repository

Install dependencies

pip install torch torchvision albumentations opencv-python scikit-learn pillow streamlit


Or use the provided requirements file:

pip install -r requirements.txt

Usage

The project consists of two main phases:

Phase 1: Dataset Preparation & Model Training (Kaggle/Colab)
Step 1: Prepare Binary Dataset

Script: prepare_binary_dataset.py

Converts original train.csv annotations into binary labels
("normal" if No finding, otherwise "abnormal")

Balances both classes

Splits into train/validation sets (80/20)


Step 2: Train the Model

Script: train_binary_resnet.py

Defines custom Albumentations dataset with augmentation

Loads ResNet-18 with ImageNet weights

Uses CrossEntropyLoss + Adam optimizer

Implements early stopping based on F1-score

Saves best model weights to:

best_classification_model.pth


Training loop metrics:

Accuracy

F1-score

ROC-AUC (for binary case)

Example:

[Val] Accuracy: 0.92, F1: 0.91, AUC: 0.94
[Save] Best model updated with F1-score: 0.91

Phase 2: Deployment (Local Machine)
Step 1: Streamlit App

Script: app.py

Loads trained model weights (best_classification_model.pth)

Allows user to upload a chest X-ray image (.jpg, .jpeg, .png)

Displays:

Predicted label (Normal / Abnormal)

Confidence percentage

Probability bar chart

Grad-CAM heatmap overlay for interpretability

Run the app:

streamlit run app_resnet_binary.py


Then open your browser at:
👉 http://localhost:8501


Link for trained weights: https://drive.google.com/file/d/1yW1qHFFwNO8BBxqoUCrFVBRrjwAqGDsJ/view?usp=drive_link

Try the website here-> https://xvc6w4cs9cvjkepo5uftgq.streamlit.app

Snapshots of the Application

![Streamlit_page-0001](https://github.com/user-attachments/assets/37755935-b54d-48e6-bf9e-4d5b1d03406f)

![Streamlit_page-0002](https://github.com/user-attachments/assets/d5402aca-67c6-40c2-a4c4-578d4c29b19d)

![Streamlit_page-0003](https://github.com/user-attachments/assets/eb5e2829-e03e-45f1-b4ee-c20e0912b324)



