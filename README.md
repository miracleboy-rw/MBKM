# Clinical Outcome Prediction System (CDSS) - Machine Learning

Repositori proyek Machine Learning untuk klasifikasi luaran klinis pasien terinfeksi bakteri (**Sembuh** vs **Meninggal**) berbasis rekam medis dan pola resistensi antibiotik.

---

## 📁 Struktur Direktori

```text
.
├── MBKM_Ivan Nur Azmi_123430119.ipynb       # Model 1: Klasifikasi Pola Resistensi Antibiotik
├── MBKM_2_Ivan_Nur_Azmi.ipynb               # Model 2: Rekam Medis & Komorbiditas Klinis
│
├── dataset/
│   ├── raw/                                 # Data mentah sumber
│   │   ├── antibiotic_resistance_tracking.csv
│   │   └── dataset_gabungan_luaran.xlsx
│   └── processed/                           # Data hasil partisi dan augmentasi
│       ├── data_latih_rekam_medis.csv       # Data latih asli (208 baris)
│       ├── data_latih_smote_rekam_medis.csv # Data latih pasca-SMOTE (350 baris)
│       └── data_uji_rekam_medis.csv         # Data uji independen (53 baris)
│
├── models/
│   └── model_luaran_rekam_medis.pkl         # Artefak model terbaik & pipeline deployment
│
└── output/                                  # Ekspor visualisasi analitis
    ├── Eksplorasi data klinis pasien.png
    ├── Penyeimbangan Kelas.png
    └── perbandingan model.png
```

---

## 🔬 Ringkasan Proyek & Model

### 1. Model 1 (`MBKM_Ivan Nur Azmi_123430119.ipynb`)
- **Dataset**: `dataset/raw/antibiotic_resistance_tracking.csv`
- **Tujuan**: Memprediksi luaran pasien berdasarkan profil resistensi 4 antibiotik utama (AMC, CIP, MEM, COL) dan tipe spesimen (*Darah, Urin, Dahak*).
- **Metodologi**: Ordinal encoding tingkat resistensi (`Sensitive=0, Intermediate=1, Resistant=2`), rekayasa fitur indeks MDR, penanganan threshold klinis Youden's J.
- **Karakteristik Data**: Data sintetis/seragam, akurasi uji berada pada batas teoritis ~53-54% (distribusi target seimbang 50:50).

### 2. Model 2 (`MBKM_2_Ivan_Nur_Azmi.ipynb`)
- **Dataset**: `dataset/raw/dataset_gabungan_luaran.xlsx` (Data Rekam Medis Nyata Rumah Sakit)
- **Tujuan**: Membangun *Clinical Decision Support System* (CDSS) memprediksi risiko mortalitas pasien rawat inap.
- **Fitur Utama**:
  - Ekstraksi NLP diagnosis/komorbiditas: `is_sepsis`, `is_kidney_failure` (CKD/AKI), `is_diabetes`, `is_stroke_ich`.
  - Indeks beban komorbiditas simultan (0–4).
  - Durasi rawat inap (`lama_rawat_num`) dan asal infeksi (*Hospital* vs *Community*).
- **Prosedur Data**:
  - Filter kriteria inklusi luaran terkonfirmasi (261 pasien: 219 Sembuh, 42 Meninggal).
  - Stratified 80:20 split sebelum imputasi median (*anti-data leakage*).
  - SMOTE khusus pada training set (175 vs 175 seimbang).
- **Benchmark 6 Model**:
  - **Gradient Boosting** (Juara): CV ROC-AUC 0.9897, Akurasi Uji 94.34% (naik ke **96.23%** pasca-Youden's J), Recall Pasien Kritis **88.89%**, ROC-AUC Uji **0.9369**.
- **Fitur Interaktif (Bagian 10)**:
  - **Versi 1**: Simulasi berbasis nomor indeks data uji.
  - **Versi 2**: GUI interaktif (`ipywidgets`) dengan form checkbox, slider, dan tombol diagnosis instan.

---

## 🚀 Panduan Menjalankan

### Prasyarat Dependensi
Pastikan Python 3.10+ terpasang, lalu instal paket yang dibutuhkan:
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn joblib openpyxl ipywidgets
```

### Menjalankan Notebook
1. Buka Jupyter Lab atau VS Code di root direktori proyek.
2. Jalankan notebook pilihan:
   - Buka `MBKM_2_Ivan_Nur_Azmi.ipynb` untuk model rekam medis.
   - Buka `MBKM_Ivan Nur Azmi_123430119.ipynb` untuk model resistensi antibiotik.
3. Jalankan seluruh sel secara berurutan (*Run All*).
4. Untuk menggunakan GUI interaktif pada Model 2, arahkan ke sel **Bagian 10 (Versi 2)**.
