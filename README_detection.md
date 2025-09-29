
Chest X-Ray Abnormality Detection (YOLOv8)

This project demonstrates a complete end-to-end pipeline for object detection on chest X-ray images. Using the VinBigData Chest X-ray dataset, a YOLOv8 model is trained to detect 14 different lung abnormalities. The trained model is then deployed via a Streamlit web application for local inference.

🚀 Project Overview

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
git clone cd

Install main ML framework
pip install ultralytics

Install other dependencies
pip install -r requirements.txt

⚙️ Usage

The project consists of two main phases:

Phase 1: Data Preparation & Training (Kaggle/Colab)

Prepare dataset in YOLO format

Configure training using vinbigdata.yaml

Train YOLOv8 model (yolov8n.pt)

Run inference on validation images

(All code and preprocessing steps are included in the notebook)

Phase 2: Deployment (Local Machine)

Download the trained weights (best.pt)

Save the deployment script as app.py

Run the Streamlit app:

streamlit run app.py

Open your browser at http://localhost:8501

Link for trained weights: https://drive.google.com/file/d/1Tt7-qfGC8509TGZTMIT_cWIovgVyRiyc/view?usp=drive_link


Snapshots of the application:
![Streamlit_page-0001](https://github.com/user-attachments/assets/c6ede700-544e-4060-a28e-5e66f04b0531)

![Streamlit_page-0002](https://github.com/user-attachments/assets/3fdd8783-1017-4738-a199-44a573767f42)

![Streamlit_page-0003](https://github.com/user-attachments/assets/cbee392f-c4f6-49c0-8f40-711d3ed22108)




