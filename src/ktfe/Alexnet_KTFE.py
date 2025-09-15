import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# === CONFIG ===
THERMAL_DATA_DIR = r"KTFE_Thermal_dataset"
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMS ===
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === LOAD DATASET ===
thermal_dataset = datasets.ImageFolder(THERMAL_DATA_DIR)
targets = [s[1] for s in thermal_dataset.samples]
subjects = [os.path.basename(s[0]).split('_')[4][:6] for s in thermal_dataset.samples]  # Extract subject ID

# === STRATIFIED SUBJECT SPLIT ===
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(sgkf.split(np.zeros(len(targets)), targets, groups=subjects))

train_dataset = Subset(thermal_dataset, train_idx)
val_dataset = Subset(thermal_dataset, val_idx)
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# === BALANCING WITH SAMPLER ===
train_labels = [targets[i] for i in train_idx]
class_counts = Counter(train_labels)
weights = [1.0 / class_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === MODEL: ALEXNET ===
model = models.alexnet(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[6] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
)

model.to(DEVICE)

# === LOSS, OPTIMIZER, LR SCHEDULER ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# === TRAINING LOOP ===
best_val_acc = 0
patience_counter = 0
train_loss_hist, val_loss_hist = [], []
train_acc_hist, val_acc_hist = [], []

for epoch in range(EPOCHS):
    model.train()
    total, correct, running_loss = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total * 100
    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)

    # === VALIDATION ===
    model.eval()
    val_loss, correct, total = 0, 0, 0
    val_preds, val_true = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    val_loss /= total
    val_acc = correct / total * 100
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({'model_state_dict': model.state_dict()}, "best_alexnet_thermal_model.pth")
        print(f"✅ Checkpoint saved at epoch {epoch+1}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⛔ Early stopping triggered.")
            break

# === PLOTTING RESULTS ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_hist, label='Train Loss')
plt.plot(val_loss_hist, label='Val Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_hist, label='Train Accuracy')
plt.plot(val_acc_hist, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

# === CONFUSION MATRIX ===
cm = confusion_matrix(val_true, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=thermal_dataset.classes)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.show()

# === CLASSIFICATION REPORT ===
print("🔍 Classification Report:\n")
print(classification_report(val_true, val_preds, target_names=thermal_dataset.classes, digits=3))
