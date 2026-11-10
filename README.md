# Image Classification on SIMPLIcity Dataset using Custom CNN

Implementasi dan eksperimen pemodelan Convolutional Neural Network (CNN) kustom dari awal (*from scratch*) untuk klasifikasi 10 kategori citra pemandangan dan objek pada SIMPLIcity Dataset (1.000 citra). Proyek ini menganalisis transisi arsitektur mulai dari penanganan *mode collapse*, eksplorasi kedalaman filter konvolusi, perbandingan mekanisme pooling spasial (*Flatten* vs *GlobalAveragePooling*), hingga perancangan arsitektur optimal **5Conv-Dense Hybrid**.

---

## Ringkasan Hasil Eksperimen

Model terbaik (**Eksperimen 3 - 5Conv-Dense Hybrid**) membukukan performa generalisasi tertinggi dan terbebas dari bias kelas:

| Metrik Evaluasi | Nilai Capaian | Keterangan |
| :--- | :--- | :--- |
| **Akurasi Validasi** | **80.67%** | Loss: 0.7251 |
| **Akurasi Pengujian (Test Accuracy)** | **73.00%** | 146 dari 200 citra uji independen tertebak benar (Loss: 0.9482) |
| **Akurasi Test (Dengan TTA)** | **73.50%** | Menerapkan *Test-Time Augmentation* (horizontal flip) |
| **Ketepatan Sampel Representatif** | **100.00% (12/12)** | Seluruh sampel visual terklasifikasi akurat dengan penanda hijau |
| **Total Parameter Bobot** | **1.247.978** | Proporsional, efisien, dan bebas *over-parameterization* |

---

## Struktur Repositori

Proyek ini telah distrukturisasi secara bersih dan modular:

```text
modeling_cnn/
├── dataset/                    # Direktori dataset citra SIMPLIcity
│   ├── train/                  # 500 citra latih (50 per kelas)
│   ├── validation/             # 300 citra validasi (30 per kelas)
│   └── test/                   # 200 citra uji independen (20 per kelas)
├── modeling/                   # Notebook eksperimen dan pelatihan model
│   ├── custom_cnn_simplicity.ipynb     # Notebook utama (5Conv-Dense Hybrid, Akurasi 73.50%)
│   └── custom_cnn_simplicity_pt2.ipynb # Eksperimen lanjutan (3-Layer Hidden Dense)
├── models/                     # Berkas bobot model terlatih format native Keras
│   └── best_custom_cnn_simplicity.keras
├── .gitignore                  # Berkas pengabaian berkas cache dan temporary
└── README.md                   # Dokumentasi resmi proyek
```

---

## Arsitektur Model Terbaik (Custom CNN 5Conv-Dense Hybrid)

Alur komputasi model dirancang bertingkat 5 tahap konvolusi dengan perataan spasial dan pengklasifikasi dense berlapis:

1. **Input Layer**: Citra RGB resolusi $128 \times 128 \times 3$.
2. **Ekstraksi Fitur Spasial (5 Blok Konvolusi)**:
   - **Blok 1**: `Conv2D(32, (3, 3), ReLU, Same)` $\rightarrow$ `MaxPooling2D(2, 2)` $\rightarrow$ Output: $64 \times 64 \times 32$
   - **Blok 2**: `Conv2D(64, (3, 3), ReLU, Same)` $\rightarrow$ `MaxPooling2D(2, 2)` $\rightarrow$ Output: $32 \times 32 \times 64$
   - **Blok 3**: `Conv2D(128, (3, 3), ReLU, Same)` $\rightarrow$ `MaxPooling2D(2, 2)` $\rightarrow$ Output: $16 \times 16 \times 128$
   - **Blok 4**: `Conv2D(256, (3, 3), ReLU, Same)` $\rightarrow$ `MaxPooling2D(2, 2)` $\rightarrow$ Output: $8 \times 8 \times 256$
   - **Blok 5**: `Conv2D(128, (3, 3), ReLU, Same)` $\rightarrow$ `MaxPooling2D(2, 2)` $\rightarrow$ Output: $4 \times 4 \times 128$
3. **Mekanisme Perataan Spasial (Flatten)**:
   - Peta fitur akhir $4 \times 4 \times 128$ diratakan menjadi **2.048 unit**, menjaga hubungan koordinat vertikal pemandangan alam (posisi langit di atas, air/tanah di bawah).
4. **Klasifikasi Dense Bertingkat**:
   - `Dense(256, ReLU)` dengan `Dropout(0.25)`
   - `Dense(128, ReLU)`
   - `Dense(10, Softmax)` untuk menghasilkan probabilitas 10 kategori target.

---

## Rincian Performa Per Kelas (200 Data Uji)

| Kategori Target | Sampel Uji | Prediksi Benar | Salah Tebak | Akurasi (%) |
| :--- | :---: | :---: | :---: | :---: |
| **dinosaur** | 20 | 20 | 0 | **100.00%** |
| **bus** | 20 | 19 | 1 | **95.00%** |
| **horse** | 20 | 19 | 1 | **95.00%** |
| **flower** | 20 | 18 | 2 | **90.00%** |
| **people** | 20 | 17 | 3 | **85.00%** |
| **building** | 20 | 17 | 3 | **85.00%** |
| **food** | 20 | 12 | 8 | **60.00%** |
| **beach** | 20 | 11 | 9 | **55.00%** |
| **mountain** | 20 | 10 | 10 | **50.00%** |
| **elephant** | 20 | 3 | 17 | **15.00%** |

---

## Petunjuk Penggunaan & Eksekusi

### 1. Kloning Repositori
```bash
git clone https://github.com/Chrissaulin/simplicity-custom-cnn-classification.git
cd simplicity-custom-cnn-classification
```

### 2. Instalasi Dependensi
Pastikan lingkungan Python 3.10+ telah terpasang, lalu jalankan:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### 3. Menjalankan Notebook
Jalankan Jupyter Notebook atau buka melalui VS Code:
```bash
jupyter notebook modeling/custom_cnn_simplicity.ipynb
```
Notebook akan secara otomatis mendeteksi direktori `dataset/` dan memuat model dari direktori `models/`.
