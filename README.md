# Facial Emotion Recognition using Thermal Images  

This repository contains source code and experimental outputs for **Facial Emotion Recognition (FER) using thermal images**. The project evaluates different deep learning models on two datasets: the **Kotani Thermal Facial Emotion (KTFE)** dataset and a **custom Proposed dataset**.  

---

## 📊 Datasets  

### 1) Kotani Thermal Facial Emotion (KTFE) Dataset  
The **KTFE dataset** is a publicly available thermal FER dataset containing **1,880 thermal images** across six basic emotions.  

- **Dataset Limitations**:  
  - Severe **class imbalance** (e.g., 480 “Sad” vs. only 20 “Angry”).  
  - High risk of **identity leakage** if subjects appear in both train and validation splits.  
  - These issues can bias learning and lead to overfitting.  

#### Class Distribution (Table I)  

| Emotion   | Images |  
|-----------|--------|  
| Angry     | 20     |  
| Disgust   | 220    |  
| Fear      | 440    |  
| Happy     | 460    |  
| Sad       | 480    |  
| Surprise  | 260    |  
| **Total** | **1880** |  

---

### 2) Proposed Dataset  
A **custom dataset** was collected containing **2,421 thermal images** of **12 subjects** expressing six core emotions: angry, disgust, fear, happy, sad, and surprise.  

- Captured in a **controlled setting** for consistency.  
- Limitation: **small number of subjects**, affecting model generalization.  

---

## 🧠 Models & Results  

| Dataset   | Model       | Accuracy |  
|-----------|------------|----------|  
| KTFE      | AlexNet     | **72.1%** |  
| KTFE      | ResNet18    | 65.0%   |  
| Proposed  | ResNet50v2  | **75.0%** |  
| Proposed  | Base CNN    | 46.3%   |  

- On **KTFE**, **AlexNet** performed best (72.1%), improved with **weighted cross-entropy** and **subject-wise stratified splits**.  
- On the **Proposed dataset**, **ResNet50v2** generalized better with 75% accuracy.  

---

## 🚀 Key Contributions  

- Designed a **subject-stratified splitting strategy** to prevent identity leakage.  
- Implemented **weighted cross-entropy** to address class imbalance.  
- Applied **data augmentation** (flips, rotations) for robustness.  
- Integrated **early stopping** to prevent overfitting.  
- Used **Grad-CAM** for model interpretability.  

---

## 📂 Repository Structure  
Facial_Emotion_Recognition/
│── code/ # Training scripts, model definitions, utilities
│── outputs/ # Logs, training curves, confusion matrices
│── README.md # Project documentation

## Requirements
torch \n
torchvision
opencv-python
numpy
matplotlib
scikit-learn



## 📈 Outputs

Training/validation loss & accuracy curves \n
Confusion matrices per dataset
