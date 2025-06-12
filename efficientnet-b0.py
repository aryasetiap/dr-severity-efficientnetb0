import os
import cv2
import numpy as np
import pandas as pd
import random
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- PENETAPAN SEED UNTUK REPRODUCIBILITY ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- KONFIGURASI PATH DATA ---
TRAIN_DIR = 'data/processed/processed_train'
VAL_DIR = 'data/processed/processed_val'
TEST_DIR = 'data/processed/processed_test'
TRAIN_CSV = 'data/train.csv'
VAL_CSV = 'data/val.csv'
TEST_CSV = 'data/test.csv'

BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_CLASSES = 4  # Severity 1-4
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EARLY_STOPPING_PATIENCE = 5

# --- Penyesuaian num_workers untuk Windows ---
import platform
NUM_WORKERS = 0 if platform.system() == "Windows" else 2

# --- DATASET TANPA PREPROCESSING BERAT (langsung load gambar hasil preprocessing) ---
class FundusProcessedDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1]) - 1  # Label severity 1-4 → 0-3
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

# --- TRANSFORMASI UNTUK TRAIN, VAL, TEST ---
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- DATASET & DATALOADER ---
train_dataset = FundusProcessedDataset(TRAIN_CSV, TRAIN_DIR, transform=train_transform)
val_dataset = FundusProcessedDataset(VAL_CSV, VAL_DIR, transform=val_test_transform)
test_dataset = FundusProcessedDataset(TEST_CSV, TEST_DIR, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- MODEL EFFICIENTNET-B0 ---
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# --- LOSS, OPTIMIZER, SCHEDULER ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# --- TRAINING LOOP DENGAN EARLY STOPPING ---
best_val_acc = 0
epochs_no_improve = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    # --- VALIDASI ---
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    scheduler.step(val_acc)

    # --- EARLY STOPPING & SIMPAN MODEL TERBAIK ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'efficientnetb0_best.pth')
        print("Model terbaik disimpan.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# --- EVALUASI AKHIR DI DATA VALIDASI ---
print("\nEvaluasi pada data validasi:")
print(classification_report(all_labels, all_preds, target_names=[f"Severity {i+1}" for i in range(NUM_CLASSES)]))
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"Severity {i+1}" for i in range(NUM_CLASSES)],
            yticklabels=[f"Severity {i+1}" for i in range(NUM_CLASSES)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Validation)')
plt.tight_layout()
plt.savefig('confusion_matrix_val.png')
plt.show()

# --- EVALUASI DI DATA TEST ---
model.load_state_dict(torch.load('efficientnetb0_best.pth'))
model.eval()
test_labels = []
test_preds = []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(predicted.cpu().numpy())

print("\nEvaluasi pada data test:")
print(classification_report(test_labels, test_preds, target_names=[f"Severity {i+1}" for i in range(NUM_CLASSES)]))
cm_test = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
            xticklabels=[f"Severity {i+1}" for i in range(NUM_CLASSES)],
            yticklabels=[f"Severity {i+1}" for i in range(NUM_CLASSES)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test)')
plt.tight_layout()
plt.savefig('confusion_matrix_test.png')
plt.show()