# Facial Emotion Recognition (FER) using Deep Learning

This repository contains experiments on **Facial Emotion Recognition** using two datasets:
1. **KTFE Dataset** – trained using AlexNet
2. **Proposed Dataset** – trained using ResNet18

We evaluated and compared the performance of different models, achieving up to **72.1% accuracy on KTFE**.

---

## 📂 Project Organization
- `datasets/` → Documentation of datasets
- `src/` → Training, evaluation, Grad-CAM, and utilities
- `models/` → Saved model checkpoints
- `results/` → Confusion matrices, Grad-CAM visualizations, learning curves
- `notebooks/` → Exploratory experiments and analysis

---

## 🔧 Features
- AlexNet & ResNet18 fine-tuning
- Weighted cross-entropy for class imbalance
- Subject-wise stratified splitting
- Early stopping & LR scheduling
- Grad-CAM visualizations for explainability

---

## 📊 Results

| Dataset    | Model    | Accuracy | Macro F1 | Weighted F1 |
|------------|----------|----------|----------|-------------|
| KTFE       | AlexNet  | 72.1%    | 0.752    | 0.714       |
| Proposed   | ResNet18 | XX%      | XX       | XX          |

*(replace XX with your actual outputs)*

---

## ⚡ Quick Start
```bash
# Clone repo
git clone https://github.com/<your-username>/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition

# Install dependencies
pip install -r requirements.txt

# Train AlexNet on KTFE
python src/ktfe/train_alexnet.py --data "path/to/KTFE"

# Train ResNet18 on Proposed Dataset
python src/proposed/train_resnet.py --data "path/to/Proposed"
