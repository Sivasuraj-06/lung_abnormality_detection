
🩻 Chest X-Ray Abnormality Detection (YOLOv8)
This project demonstrates a complete end-to-end pipeline for Object Detection on medical images. It uses the VinBigData Chest X-ray dataset to train a YOLOv8 model to detect 14 different lung abnormalities, followed by deployment of the trained model via a Streamlit web application.

🚀 Project Overview
The core goal of this project is to accurately localize and classify various lung abnormalities in chest X-ray images. The workflow covers data cleaning, feature engineering, custom data splitting using GroupKFold, model training on a remote server (Kaggle/Colab), and finally, local deployment using a simple Python script and Streamlit.

Feature	Details
Model	YOLOv8n (Nano)
Dataset	VinBigData Chest X-ray (Filtered)
Task	Object Detection
Training Env	Online IDE (Kaggle/Colab)
Deployment	Streamlit (Local Host)
Image Size	640x640 pixels

🛠️ Technologies & Dependencies
The project uses a standard Python data science stack with the Ultralytics library for training and inference.

Python: 3.8+

Deep Learning: torch, ultralytics (YOLOv8)

Data Science: numpy, pandas, sklearn, tqdm

Visualization: matplotlib, seaborn

Deployment: streamlit, Pillow (PIL)

Utilities: os, shutil, glob, yaml, cv2

Installation
First, clone the repository and install the main requirements.

Bash

# Clone the repository (assuming your deployment code is in app.py)
git clone <your-repo-link>
cd <your-repo-name>

# Install the necessary libraries
!pip install ultralytics  # Main ML framework
pip install -r requirements.txt # (Requires other libraries to be listed here)
⚙️ Usage: The Workflow
The project is structured into two main phases: Data Preparation & Training (performed on the online IDE, typically a Jupyter Notebook) and Deployment (performed locally).

Phase 1: Data Preparation and Training (Online IDE/Kaggle)
These steps are executed sequentially in your notebook (.ipynb file) to prepare the data, configure the environment, and start training.

1. Imports and Configuration
Initialize the environment and define constants.

Python

# Imports and Initial Setup
import numpy as np, pandas as pd
from glob import glob
import shutil, os
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from tqdm.notebook import tqdm
import seaborn as sns
import ultralytics
import torch
from IPython.display import clear_output 
from os import listdir
from os.path import isfile, join
import yaml

# Configuration Variables
dim = 512  # Feature/Image Dimension
fold = 4   # Initial fold variable (note: GroupKFold uses n_splits=5 later)

# Environment Check
clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
ultralytics.checks()
2. Data Loading and Cleaning
Load the CSV metadata, construct dynamic image paths, and perform initial filtering.

Python

# Load Data and Create Image Paths
train_df = pd.read_csv(f'../input/vinbigdata-{dim}-image-dataset/vinbigdata/train.csv')
train_df['image_path'] = f'/kaggle/input/vinbigdata-{dim}-image-dataset/vinbigdata/train/'+train_df.image_id+('.png' if dim!='original' else '.jpg')

# Filter out the 'No Finding' class (class_id=14) as it's not needed for object detection
train_df = train_df[train_df.class_id!=14].reset_index(drop = True)
3. Bounding Box Normalization and Feature Definition
Normalize absolute pixel coordinates to a 0-1 range and define target features.

Python

# Normalize and Calculate Bounding Box Features (x/y/w/h/area in 0-1 range)
train_df['x_min'] = train_df.apply(lambda row: (row.x_min)/row.width, axis =1)
train_df['y_min'] = train_df.apply(lambda row: (row.y_min)/row.height, axis =1)
train_df['x_max'] = train_df.apply(lambda row: (row.x_max)/row.width, axis =1)
train_df['y_max'] = train_df.apply(lambda row: (row.y_max)/row.height, axis =1)
train_df['x_mid'] = train_df.apply(lambda row: (row.x_max+row.x_min)/2, axis =1)
train_df['y_mid'] = train_df.apply(lambda row: (row.y_max+row.y_min)/2, axis =1)
train_df['w'] = train_df.apply(lambda row: (row.x_max-row.x_min), axis =1)
train_df['h'] = train_df.apply(lambda row: (row.y_max-row.y_min), axis =1)
train_df['area'] = train_df['w']*train_df['h']

# Define Features and Target (for potential model or analysis)
features = ['x_min', 'y_min', 'x_max', 'y_max', 'x_mid', 'y_mid', 'w', 'h', 'area']
X = train_df[features]
y = train_df['class_id']
4. Cross-Validation and File Separation
Use GroupKFold to ensure image integrity across train/val splits, then separate the file paths for a single fold.

Python

# Extract and Sort Class Labels (used for YOLO configuration)
class_ids, class_names = list(zip(*set(zip(train_df.class_id, train_df.class_name))))
classes = list(np.array(class_names)[np.argsort(class_ids)])
classes = list(map(lambda x: str(x), classes))

# GroupKFold Split (using image_id as the group)
gkf  = GroupKFold(n_splits = 5)
train_df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups = train_df.image_id.tolist())):
    train_df.loc[val_idx, 'fold'] = fold

# Separate unique image paths for the selected fold (using the defined 'fold' variable)
train_files = []
val_files   = []
val_files += list(train_df[train_df.fold==fold].image_path.unique())
train_files += list(train_df[train_df.fold!=fold].image_path.unique())
5. Directory Setup and File Copying (YOLO Format)
Create the standard YOLO directory structure and copy the images (.png / .jpg) and pre-generated YOLO label files (.txt).

Python

# Create YOLO Directory Structure
os.makedirs('/kaggle/working/vinbigdata/labels/train', exist_ok = True)
os.makedirs('/kaggle/working/vinbigdata/labels/val', exist_ok = True)
os.makedirs('/kaggle/working/vinbigdata/images/train', exist_ok = True)
os.makedirs('/kaggle/working/vinbigdata/images/val', exist_ok = True)

# Copy Images and Corresponding Label Files
label_dir = '/kaggle/input/vinbigdata-yolo-labels-dataset/labels'
for file in tqdm(train_files):
    shutil.copy(file, '/kaggle/working/vinbigdata/images/train')
    filename = file.split('/')[-1].split('.')[0]
    shutil.copy(os.path.join(label_dir, filename+'.txt'), '/kaggle/working/vinbigdata/labels/train')
    
for file in tqdm(val_files):
    shutil.copy(file, '/kaggle/working/vinbigdata/images/val')
    filename = file.split('/')[-1].split('.')[0]
    shutil.copy(os.path.join(label_dir, filename+'.txt'), '/kaggle/working/vinbigdata/labels/val')
6. Generate YOLO Configuration (.yaml)
Create the necessary configuration file that links the data paths and class information for the YOLO training command.

Python

cwd = '/kaggle/working/'

# Create train.txt and val.txt with absolute image paths
with open(join( cwd , 'train.txt'), 'w') as f:
    for path in glob('/kaggle/working/vinbigdata/images/train/*'):
        f.write(path+'\n')
            
with open(join( cwd , 'val.txt'), 'w') as f:
    for path in glob('/kaggle/working/vinbigdata/images/val/*'):
        f.write(path+'\n')

# Define YOLO config content (nc=14 for the filtered classes)
data = dict(
    train =  join( cwd , 'train.txt') ,
    val   =  join( cwd , 'val.txt' ),
    nc    = 14,
    names = classes
    )

# Write to vinbigdata.yaml
with open(join( cwd , 'vinbigdata.yaml'), 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)
7. Training and Inference Commands
Run the training, check results, and run inference on the validation set. The best.pt file from the training run is the model that needs to be downloaded for deployment.

Bash

# --- Training Command ---
# Train YOLOv8 Nano model for 100 epochs
!yolo train model=yolov8n.pt workers=8 device=0 batch=32 data = /kaggle/working/vinbigdata.yaml imgsz = 640 epochs = 100

# --- Training Sanity Checks (Visualization) ---
# View summary of labels used in training
plt.figure(figsize = (20,20))
plt.axis('off')
plt.imshow(plt.imread('runs/detect/train/labels.jpg'));

# View first few training batches (with ground truth labels)
plt.figure(figsize = (15, 15))
plt.imshow(plt.imread('runs/detect/train/train_batch0.jpg'))
# ... train_batch1.jpg, train_batch2.jpg

# --- Inference Command ---
# Run prediction on the validation set using the best weights
!yolo predict model = runs/detect/train/weights/best.pt\
imgsz = 640\
conf = 0.25\
iou = 0.5\
source = /kaggle/working/vinbigdata/images/val

# --- Final Visualization of Predictions ---
# Visualize a grid of random prediction results from the 'runs/detect/predict' folder
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# ... (full visualization code block)
Phase 2: Deployment (Local Machine)
The final step is to create the deployment app.

Download Model: Download the trained weights file, typically located at runs/detect/train/weights/best.pt, from your online IDE to your local machine.

Save Deployment Code: Save the following snippet as a Python file (e.g., app.py).

Run Streamlit: Execute the file from your terminal.

app.py
Python

import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ---------------------------
# Model and Classes Configuration
# ---------------------------
# Caching the model ensures fast reloads
@st.cache_resource
def load_model():
    # Model weights must be in the same directory as this file
    model = YOLO("best.pt")
    return model

model = load_model()

# Use your dataset class names (14 classes excluding 'No Finding')
class_names = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"
]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🩻 Chest X-Ray Abnormality Detection (YOLO)")
st.write("Upload a Chest X-ray image and the model will detect abnormalities.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO inference
    results = model.predict(image, imgsz=640, conf=0.25)
    res = results[0]

    # Show annotated image (with bounding boxes)
    annotated = res.plot() 
    st.image(annotated, caption="Detections", use_column_width=True)

    # Show detection details
    if len(res.boxes) > 0:
        st.write("### Detected Findings")
        det_summary = {}
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = class_names[cls_id]
            st.write(f"- {cls_name} **({conf:.2f})**")
            det_summary[cls_name] = max(det_summary.get(cls_name, 0), conf)

        # Show confidence bar chart
        st.write("### Confidence Scores")
        st.bar_chart(det_summary)
    else:
        st.success("✅ No abnormalities detected")
Running the App
Navigate to the directory containing app.py and best_100.pt and run:

Bash

streamlit run app.py
This will launch the application in your local web browser, usually at http://localhost:8501.
