# =============================================================================
# BAGIAN 1: PRA-PEMROSESAN DATA (dari preprocessing.py)
# =============================================================================
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import platform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --- KONFIGURASI PATH DATA & PARAMETER ---
# Path
IMG_DIR = 'data/raw'
CSV_PATH = 'data/images_id_kelas.csv'
PREP_TRAIN_DIR = 'data/processed/processed_train'
PREP_VAL_DIR = 'data/processed/processed_val'
PREP_TEST_DIR = 'data/processed/processed_test'
TRAIN_CSV = 'data/train.csv'
VAL_CSV = 'data/val.csv'
TEST_CSV = 'data/test.csv'

# Parameter Model & Pelatihan
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_CLASSES = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0 if platform.system() == "Windows" else 2
SEED = 42

# --- FUNGSI UNTUK REPRODUCIBILITY ---
def set_seed(seed):
    """Menetapkan seed untuk semua library yang relevan."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- FUNGSI-FUNGSI PRA-PEMROSESAN ---
def apply_clahe(img):
    """Menerapkan CLAHE pada channel L dari gambar LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def apply_sharpening(img):
    """Menerapkan filter sharpening pada gambar."""
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    return cv2.filter2D(img, -1, kernel)

def preprocess_and_save(df, out_dir):
    """Fungsi utama untuk melakukan pra-pemrosesan dan menyimpan gambar."""
    os.makedirs(out_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing to {out_dir}"):
        img_name = row.iloc[0]
        img_path = os.path.join(IMG_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image not found: {img_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        green_channel = image[:, :, 1]
        image = np.stack([green_channel] * 3, axis=-1)
        image = apply_clahe(image)
        image = apply_sharpening(image)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def run_preprocessing_pipeline():
    """Menjalankan seluruh pipeline pra-pemrosesan: split, proses, simpan."""
    print("Memulai pipeline pra-pemrosesan...")
    df = pd.read_csv(CSV_PATH)
    
    # Split dataset
    print("Membagi dataset (80% train, 10% val, 10% test)...")
    train_df = df.sample(frac=0.8, random_state=SEED)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=SEED)
    test_df = temp_df.drop(val_df.index)
    
    # Simpan split ke CSV
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    
    # Proses dan simpan gambar
    preprocess_and_save(train_df, PREP_TRAIN_DIR)
    preprocess_and_save(val_df, PREP_VAL_DIR)
    preprocess_and_save(test_df, PREP_TEST_DIR)
    
    print("Pipeline pra-pemrosesan selesai.")


# =============================================================================
# BAGIAN 2: PELATIHAN & EVALUASI MODEL (dari efficientnet-b0.ipynb)
# =============================================================================

# --- KELAS DATASET PYTORCH ---
class FundusProcessedDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

def run_training_and_evaluation():
    """Menjalankan pipeline pelatihan dan evaluasi model."""
    print("\nMemulai pipeline pelatihan dan evaluasi...")
    set_seed(SEED)

    # --- TRANSFORMASI DAN DATALOADER ---
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

    train_dataset = FundusProcessedDataset(TRAIN_CSV, PREP_TRAIN_DIR, transform=train_transform)
    val_dataset = FundusProcessedDataset(VAL_CSV, PREP_VAL_DIR, transform=val_test_transform)
    test_dataset = FundusProcessedDataset(TEST_CSV, PREP_TEST_DIR, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- MODEL, LOSS, OPTIMIZER, SCHEDULER ---
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # --- TRAINING LOOP ---
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0
    epochs_no_improve = 0

    print("Memulai pelatihan...")
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
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # --- VALIDASI ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        scheduler.step(val_acc)

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
    
    print("Pelatihan selesai.")

    # --- VISUALISASI HASIL PELATIHAN ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.show()

    # --- EVALUASI PADA DATA TEST ---
    print("\nMemulai evaluasi pada data test...")
    model.load_state_dict(torch.load('efficientnetb0_best.pth', map_location=DEVICE))
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

    print("\nLaporan Klasifikasi pada Data Test:")
    print(classification_report(test_labels, test_preds, target_names=[f"Severity {i}" for i in range(NUM_CLASSES)], zero_division=0))
    
    cm_test = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
                xticklabels=[f"Severity {i}" for i in range(NUM_CLASSES)],
                yticklabels=[f"Severity {i}" for i in range(NUM_CLASSES)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test)')
    plt.tight_layout()
    plt.show()


# =============================================================================
# --- MAIN EXECUTION BLOCK ---
# =============================================================================
if __name__ == '__main__':
    # Langkah 1: Jalankan pra-pemrosesan.
    # Cukup jalankan sekali saja. Jika data 'processed' sudah ada, baris ini bisa di-comment.
    # run_preprocessing_pipeline()

    # Langkah 2: Jalankan pelatihan dan evaluasi model.
    run_training_and_evaluation()