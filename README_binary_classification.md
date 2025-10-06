# CliniScan: Lung-Abnormality Classification on Chest X-rays (ResNet-18) 

This project demonstrates a complete end-to-end pipeline for **binary classification** on Chest X-ray (CXR) images. Using the **VinBigData Chest X-ray** dataset, a **ResNet-18** model is trained to classify the presence of lung abnormalities as either **Abnormal** or **Normal**.

The trained model is then deployed via a **Streamlit** web application, featuring **Grad-CAM** explainability for enhanced clinical interpretability.



##  Project Overview

The primary goal is to accurately classify a single chest X-ray as either **Normal** or **Abnormal**.

### Workflow Highlights:
* **Binary Dataset Preparation and Balancing:** Converting multi-label annotations into a balanced binary target.
* **Training:** Using PyTorch with a pre-trained **ResNet-18** backbone.
* **Deployment:** Interactive web application using Streamlit.
* **Interpretability:** **Grad-CAM** visualizations to highlight decision-making regions.

###  Feature Details

| Feature | Description |
| :--- | :--- |
| **Model** | **ResNet-18** (Pre-trained on ImageNet) |
| **Dataset** | VinBigData Chest X-ray (Binary Targets) |
| **Task** | **Binary Image Classification** (Normal / Abnormal) |
| **Training Env** | Kaggle / Colab (PyTorch) / Local Machine|
| **Deployment** | Streamlit (Local Host) |




##  Technologies & Dependencies

* **Python:** $3.8+$
* **Deep Learning:** `torch`, `torchvision`
* **Data Augmentation:** `albumentations`, `opencv-python`
* **Metrics:** `scikit-learn`
* **Image Handling:** `Pillow` (PIL)
* **Deployment:** `streamlit`

### Installation

1.  **Clone the repository** (if applicable).
2.  **Install dependencies:**
    ```bash
    pip install torch torchvision albumentations opencv-python scikit-learn pillow streamlit
    ```
    *Or use the provided requirements file:*
    ```bash
    pip install -r requirements.txt
    ```



##  Usage

The project consists of two main phases: Training and Deployment.

### Phase 1: Dataset Preparation & Model Training (Kaggle/Colab)

**Step 1: Prepare Binary Dataset**
* **Script:** `prepare_binary_dataset.py`
* Converts original `train.csv` annotations into binary labels: ("normal" if 'No finding', otherwise "abnormal").
* Balances both classes and splits into train/validation sets ($80/20$).

**Step 2: Train the Model**
* **Script:** `train_binary_resnet.py`
* Defines a custom **Albumentations** dataset with augmentation.
* Loads **ResNet-18** with ImageNet pre-trained weights.
* Uses **CrossEntropyLoss** and **Adam optimizer**.
* Implements **early stopping** based on the $\text{F1-score}$.
* Saves best model weights to: `best_classification_model.pth`

**Example Training Loop Metrics:**
[Val] Accuracy: 0.92, F1: 0.91, AUC: 0.94
[Save] Best model updated with F1-score: 0.91

### Phase 2: Deployment (Local Machine)

**Step 1: Streamlit App Setup**
* **Model Weights:** The trained weights (`best_classification_model.pth`) are available for download here: [Trained Weights Link](https://drive.google.com/file/d/1yW1qHFFwNO8BBxqoUCrFVBRrjwAqGDsJ/view?usp=drive_link)
* **Deployment Script:** Save the deployment script as `app.py` or use the provided file (`app_resnet_binary.py`).

**Step 2: Run the Application**
```bash
streamlit run app_resnet_binary.py
Open your browser at:  http://localhost:8501
```



**Snapshots of the Application**

![Streamlit_page-0001](https://github.com/user-attachments/assets/37755935-b54d-48e6-bf9e-4d5b1d03406f)

![Streamlit_page-0002](https://github.com/user-attachments/assets/d5402aca-67c6-40c2-a4c4-578d4c29b19d)

![Streamlit_page-0003](https://github.com/user-attachments/assets/eb5e2829-e03e-45f1-b4ee-c20e0912b324)



