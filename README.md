# 👁️ Deteksi Keparahan Diabetic Retinopathy (EfficientNet-B0)

**Oleh: Arya Setia Pratama**

Proyek ini bertujuan membangun sistem klasifikasi tingkat keparahan _Diabetic Retinopathy_ (DR) dari citra fundus retina menggunakan model _Deep Learning_ **EfficientNet-B0** dengan pendekatan _transfer learning_.

---

## 📂 Struktur Proyek

```
dr-severity-efficientnetb0/
│
├── efficientnet-b0.ipynb         # Notebook utama untuk training & evaluasi
├── preprocessing.py              # Skrip untuk pra-pemrosesan gambar
├── requirements.txt              # Daftar dependensi library
├── efficientnetb0_best.pth       # File bobot model terbaik (hasil training)
├── Laporan_Akhir_CV.pdf          # (Opsional) Laporan akhir proyek
├── docs/                         # (Opsional) Folder untuk gambar README
│   ├── sample_preprocessing.png
│   ├── loss_acc_curve.png
│   └── confusion_matrix.png
├── data/
│   ├── images_id_kelas.csv       # File CSV utama (image_id, diagnosis)
│   ├── train.csv                 # Data training (setelah split)
│   ├── val.csv                   # Data validasi (setelah split)
│   ├── test.csv                  # Data test (setelah split)
│   ├── raw/                      # Folder gambar asli
│   │   └── *.png
│   └── processed/                # Folder hasil preprocessing
│       ├── processed_train/
│       ├── processed_val/
│       └── processed_test/
└── README.md
```

---

## 📝 Deskripsi Proyek

Pipeline _end-to-end_ untuk _computer vision_ pada citra medis, meliputi:

- **Pra-pemrosesan Gambar:** Ekstraksi kanal hijau, CLAHE, sharpening untuk menonjolkan fitur patologis.
- **Pembagian Data:** Split acak per kelas (80% train, 10% val, 10% test), balancing & augmentasi per split.
- **Pelatihan Model:** Transfer learning EfficientNet-B0 (pretrained ImageNet), fine-tuning, dropout, scheduler CosineAnnealingLR, early stopping.
- **Evaluasi & Analisis:** Kurva loss/accuracy, confusion matrix, macro F1-score, visualisasi hasil, analisis performa tiap kelas.

---

## 🚀 Cara Menjalankan

### 1. Persiapan Awal

- Clone repository ini:
  ```bash
  git clone https://github.com/aryasetiap/dr-severity-efficientnetb0.git
  cd dr-severity-efficientnetb0
  ```
- Buat environment dan install dependensi:
  ```bash
  python -m venv venv
  # Aktifkan venv:
  # Windows:
  venv\Scripts\activate
  # Linux/Mac:
  source venv/bin/activate
  pip install -r requirements.txt
  ```

### 2. Siapkan Data

- Letakkan semua gambar citra fundus asli di folder `data/raw/`.
- Pastikan file `data/images_id_kelas.csv` ada, berisi dua kolom: `image_id` (nama file gambar) dan `diagnosis` (label kelas 0-4).

### 3. Pra-pemrosesan Data

- Jalankan:
  ```bash
  python preprocessing.py
  ```
- Akan dihasilkan file `train.csv`, `val.csv`, `test.csv` dan folder processed.

### 4. Training & Evaluasi

- Buka dan jalankan notebook `efficientnet-b0.ipynb`.
- Notebook akan melakukan training, validasi, simpan model terbaik (`efficientnetb0_best.pth`), dan evaluasi pada data test.

---

## 📊 Contoh Hasil Visualisasi

- Baris atas: citra asli, baris bawah: citra setelah preprocessing.
- Kurva loss/accuracy menunjukkan tren overfitting yang berhasil diatasi dengan early stopping.
- Confusion matrix dan metrik makro F1-score untuk evaluasi adil antar kelas.

---

## ⚙️ Fitur & Teknik yang Digunakan

- **Pra-pemrosesan:** Ekstraksi kanal hijau, CLAHE, sharpening, resize.
- **Balancing Data:** Oversampling/augmentasi minoritas per split.
- **Model:** EfficientNet-B0 (PyTorch, transfer learning, fine-tuning layer akhir, dropout).
- **Training:**
  - Augmentasi variatif (Albumentations, CoarseDropout, dsb)
  - Scheduler CosineAnnealingLR
  - Early stopping (patience besar)
  - Loss: CrossEntropy & Focal Loss (eksperimen)
- **Evaluasi:** Macro F1-score, confusion matrix, akurasi, presisi, recall, spesifisitas, visualisasi kurva & hasil prediksi.

---

## 📚 Referensi

- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
- Decenciere, E., et al. (2014). Feedback on a publicly available image database: the Messidor database. Image Analysis & Stereology.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. J. Artif. Intell. Res.
- PyTorch, Albumentations, scikit-learn documentation.

---

## 🧑‍💻 Kontribusi

Kontribusi sangat terbuka! Silakan fork repo ini, buat branch baru, dan ajukan pull request. Untuk diskusi atau laporan bug, silakan buat Issue baru.

---

## 🙏 Terima Kasih

Proyek ini dibuat sebagai pemenuhan tugas akhir mata kuliah Computer Vision di Universitas Lampung oleh Arya Setia Pratama. Jangan lupa ⭐ repo ini jika Anda merasa terbantu!

**Happy Coding!** 😎
