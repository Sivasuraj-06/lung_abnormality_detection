
Chest X-Ray Abnormality Detection (YOLOv8)

This project demonstrates a complete end-to-end pipeline for object detection on chest X-ray images. Using the VinBigData Chest X-ray dataset, a YOLOv8 model is trained to detect 14 different lung abnormalities. The trained model is then deployed via a Streamlit web application for local inference.

Project Overview

The project workflow includes:

Data cleaning & preprocessing

Feature engineering

Custom train/validation split using GroupKFold

Model training on Kaggle/Colab

Deployment using a Streamlit web app

Feature Details Model YOLOv8n (Nano) Dataset VinBigData Chest X-ray (Filtered) Task Object Detection Training Env Online IDE (Kaggle/Colab) Deployment Streamlit (Local Host) Image Size 640x640 pixels 🛠️ Technologies & Dependencies

Python: 3.8+

Deep Learning: torch, ultralytics (YOLOv8)

Data Science: numpy, pandas, sklearn, tqdm

Visualization: matplotlib, seaborn

Deployment: streamlit, Pillow (PIL)

Utilities: os, shutil, glob, yaml, cv2

Installation

Clone the repository

Install main ML framework
pip install ultralytics

Install other dependencies
pip install -r requirements.txt

Usage

The project consists of two main phases:

Phase 1: Data Preparation & Training (Kaggle/Colab)

Prepare dataset in YOLO format

Configure training using vinbigdata.yaml

Train YOLOv8 model (yolov8n.pt)

Run inference on validation images

(All code and preprocessing steps are included in the notebook)

Phase 2: Deployment (Local Machine)

Download the trained weights (best.pt)

The model weights (best.pt) are hosted on Google Drive and will be automatically downloaded by the Streamlit app.

Save the deployment script as app.py

Run the Streamlit app:

streamlit run app.py

Open your browser at http://localhost:8501

Link for trained weights: https://drive.google.com/file/d/1Tt7-qfGC8509TGZTMIT_cWIovgVyRiyc/view?usp=drive_link

Try the website here-> https://8dswaderpka4ehefeifmuw.streamlit.app


Snapshots of the application:
![Streamlit_d_page-0001](https://github.com/user-attachments/assets/f7fe628e-54df-43bb-b4f0-34ac558c9188)
![Streamlit_d_page-0002](https://github.com/user-attachments/assets/4adeba2c-89a9-4cf5-ba90-29194ef841bf)
![Streamlit_d_page-0003](https://github.com/user-attachments/assets/650cfb56-23f5-4fce-8f84-51c1b5049190)







