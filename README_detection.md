
🩻 Chest X-Ray Abnormality Detection (YOLOv8)

This project demonstrates a complete end-to-end pipeline for object detection on chest X-ray images. Using the VinBigData Chest X-ray dataset, a YOLOv8 model is trained to detect 14 different lung abnormalities. The trained model is then deployed via a Streamlit web application for local inference.

🚀 Project Overview

The goal of this project is to accurately localize and classify lung abnormalities in chest X-ray images. The workflow includes:

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

Clone the repository and install dependencies:

Clone the repository
git clone cd

Install main ML framework
pip install ultralytics

Install other dependencies
pip install -r requirements.txt

⚙️ Usage Workflow

The project has two main phases: Data Preparation & Training (online) and Deployment (local machine).

Phase 1: Data Preparation & Training (Kaggle/Colab)

Imports & Configuration import numpy as np, pandas as pd from glob import glob import shutil, os import matplotlib.pyplot as plt from sklearn.modelselection import GroupKFold from tqdm.notebook import tqdm import seaborn as sns import ultralytics import torch from IPython.display import clearoutput import yaml
Configuration
dim = 512 # Image dimension fold = 4 # Fold for validation split

Environment check
clearoutput() print('Setup complete. Using torch %s %s' % (torch.version, torch.cuda.getdeviceproperties(0) if torch.cuda.isavailable() else 'CPU')) ultralytics.checks()

Data Loading & Cleaning traindf = pd.readcsv(f'../input/vinbigdata-{dim}-image-dataset/vinbigdata/train.csv') traindf['imagepath'] = f'/kaggle/input/vinbigdata-{dim}-image-dataset/vinbigdata/train/'+traindf.imageid+('.png' if dim!='original' else '.jpg')
Filter out 'No Finding' class (class_id=14)
traindf = traindf[traindf.classid != 14].reset_index(drop=True)

Bounding Box Normalization traindf['xmin'] = traindf.xmin / traindf.width traindf['ymin'] = traindf.ymin / traindf.height traindf['xmax'] = traindf.xmax / traindf.width traindf['ymax'] = traindf.ymax / traindf.height traindf['xmid'] = (traindf.xmin + traindf.xmax) / 2 traindf['ymid'] = (traindf.ymin + traindf.ymax) / 2 traindf['w'] = traindf.xmax - traindf.xmin traindf['h'] = traindf.ymax - traindf.ymin traindf['area'] = traindf.w * train_df.h

Cross-Validation & File Separation

Extract class names
classids, classnames = list(zip(*set(zip(traindf.classid, traindf.classname)))) classes = list(np.array(classnames)[np.argsort(classids)]) classes = list(map(str, classes))

GroupKFold
gkf = GroupKFold(nsplits=5) traindf['fold'] = -1 for foldid, (trainidx, validx) in enumerate(gkf.split(traindf, groups=traindf.imageid.tolist())): traindf.loc[validx, 'fold'] = fold_id

Separate train/val files
trainfiles = list(traindf[traindf.fold != fold].imagepath.unique()) valfiles = list(traindf[traindf.fold == fold].imagepath.unique())

Directory Setup & File Copying (YOLO Format) os.makedirs('/kaggle/working/vinbigdata/labels/train', existok=True) os.makedirs('/kaggle/working/vinbigdata/labels/val', existok=True) os.makedirs('/kaggle/working/vinbigdata/images/train', existok=True) os.makedirs('/kaggle/working/vinbigdata/images/val', existok=True)
label_dir = '/kaggle/input/vinbigdata-yolo-labels-dataset/labels'

for file in trainfiles: shutil.copy(file, '/kaggle/working/vinbigdata/images/train') filename = file.split('/')[-1].split('.')[0] shutil.copy(os.path.join(labeldir, filename+'.txt'), '/kaggle/working/vinbigdata/labels/train')

for file in valfiles: shutil.copy(file, '/kaggle/working/vinbigdata/images/val') filename = file.split('/')[-1].split('.')[0] shutil.copy(os.path.join(labeldir, filename+'.txt'), '/kaggle/working/vinbigdata/labels/val')

Generate YOLO Configuration cwd = '/kaggle/working/'
Create train.txt and val.txt
with open(os.path.join(cwd, 'train.txt'), 'w') as f: for path in glob('/kaggle/working/vinbigdata/images/train/*'): f.write(path+'\n')

with open(os.path.join(cwd, 'val.txt'), 'w') as f: for path in glob('/kaggle/working/vinbigdata/images/val/*'): f.write(path+'\n')

YOLO YAML config
data = dict( train=os.path.join(cwd, 'train.txt'), val=os.path.join(cwd, 'val.txt'), nc=14, names=classes )

with open(os.path.join(cwd, 'vinbigdata.yaml'), 'w') as outfile: yaml.dump(data, outfile, defaultflowstyle=False)

Training & Inference
Train YOLOv8 Nano model
!yolo train model=yolov8n.pt workers=8 device=0 batch=32 data=/kaggle/working/vinbigdata.yaml imgsz=640 epochs=100

Predict on validation images
!yolo predict model=runs/detect/train/weights/best.pt imgsz=640 conf=0.25 iou=0.5 source=/kaggle/working/vinbigdata/images/val

Phase 2: Deployment (Local Machine)

Download the trained weights: best.pt

Save deployment code as app.py.

import streamlit as st from ultralytics import YOLO from PIL import Image

@st.cacheresource def loadmodel(): return YOLO("best.pt")

model = load_model()

class_names = [ "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis" ]

st.title("🩻 Chest X-Ray Abnormality Detection (YOLO)") st.write("Upload a Chest X-ray image and the model will detect abnormalities.")

uploadedfile = st.fileuploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploadedfile: image = Image.open(uploadedfile).convert("RGB") st.image(image, caption="Uploaded Image", usecolumnwidth=True)

results = model.predict(image, imgsz=640, conf=0.25)
res = results[0]

annotated = res.plot()
st.image(annotated, caption="Detections", use_column_width=True)

if len(res.boxes) > 0:
    st.write("### Detected Findings")
    det_summary = {}
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        cls_name = class_names[cls_id]
        st.write(f"- {cls_name} **({conf:.2f})**")
        det_summary[cls_name] = max(det_summary.get(cls_name, 0), conf)
    st.write("### Confidence Scores")
    st.bar_chart(det_summary)
else:
    st.success("✅ No abnormalities detected")
Run the Streamlit App streamlit run app.py

Open your browser at http://localhost:8501 to use the app.


Snapshots of the application:
![Streamlit_page-0001](https://github.com/user-attachments/assets/c6ede700-544e-4060-a28e-5e66f04b0531)

![Streamlit_page-0002](https://github.com/user-attachments/assets/3fdd8783-1017-4738-a199-44a573767f42)

![Streamlit_page-0003](https://github.com/user-attachments/assets/cbee392f-c4f6-49c0-8f40-711d3ed22108)




