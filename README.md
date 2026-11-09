# Clinical Outcome Prediction System (CDSS) - Machine Learning

Repositori proyek Machine Learning untuk stratifikasi risiko mortalitas dan klasifikasi luaran klinis pasien terinfeksi bakteri (**Sembuh** vs **Meninggal**) berbasis rekam medis dan profil mikrobiologi.

---

## 📁 Struktur Direktori

```text
.
├── MBKM_Ivan Nur Azmi_123430119.ipynb       # Model 1: Klasifikasi Pola Resistensi Antibiotik
├── MBKM_2_Ivan_Nur_Azmi.ipynb               # Model 2: Rekam Medis & Komorbiditas Klinis (CDSS)
├── clinical_preprocessor.py                 # Modul kustom pipeline inferensi rekam medis
│
├── dataset/
│   ├── raw/                                 # Data mentah sumber
│   │   ├── antibiotic_resistance_tracking.csv
│   │   └── dataset_gabungan_luaran.xlsx
│   └── processed/                           # Data terproses (19 fitur + patient_id + target)
│       ├── data_latih_rekam_medis.csv       # Data latih asli (192 baris, 66 pasien)
│       ├── data_latih_standard_cost_sensitive.csv # Data latih berbobot cost-sensitive (192 baris)
│       └── data_uji_rekam_medis.csv         # Data uji independen (31 baris, 17 pasien)
│
├── models/
│   ├── model_luaran_rekam_medis.pkl         # Artefak pipeline Gradient Boosting mandiri
│   └── pipeline_stratifikasi_risiko_pycaret.pkl
│
└── output/                                  # Visualisasi diagnostik & biostatistik
    ├── persentase_missing_values.png        # Audit kelengkapan data
    ├── Eksplorasi data klinis pasien.png    # Distribusi komorbiditas & patogen
    ├── forest_plot_rr.png                   # Forest plot Relative Risk (Isolat vs Pasien)
    ├── penyeimbangan_kelas.png              # Distribusi bobot efektif cost-sensitive
    ├── perbandingan_model.png               # Komparasi 7 algoritma Repeated CV & Brier score
    └── diagnostik_model_terpilih.png        # Confusion Matrix, ROC Pasien (n=17), Feature Importance
```

---

## 🔬 Ringkasan Proyek & Metodologi

### 1. Model 1: Pola Resistensi Antibiotik (`MBKM_Ivan Nur Azmi_123430119.ipynb`)
- **Dataset**: `dataset/raw/antibiotic_resistance_tracking.csv`
- **Fokus**: Memprediksi luaran pasien berdasarkan profil kepekaan 4 antibiotik utama (AMC, CIP, MEM, COL) dan tipe spesimen (*Darah, Urin, Dahak*).
- **Metodologi**: Ordinal encoding resistensi (`Sensitive=0, Intermediate=1, Resistant=2`), ekstraksi skor resistensi majemuk, kalibrasi threshold klinis Youden's J.

---

### 2. Model 2: CDSS Stratifikasi Risiko Mortalitas (`MBKM_2_Ivan_Nur_Azmi.ipynb`)
- **Dataset**: `dataset/raw/dataset_gabungan_luaran.xlsx` (Rekam Medis Infeksi Bakteri RS)
- **Ukuran Sampel Efektif**: 223 baris isolat kultur dari **83 profil admisi pasien unik** (67 Sembuh, 16 Meninggal).
- **Eliminasi Bias & Confounding Administratif**:
  - Kolom `Lama rawat inap` dikeluarkan untuk mencegah *target leakage*.
  - Fitur `is_hypertension`, `is_hospital_origin`, dan `riwayat_ab` dieliminasi total karena bertindak sebagai *proxy* artefak pencatatan administratif antara kohort ID 7-digit vs Kode D.
  - Model murni menggunakan **7 sindrom klinis objektif** (`is_respiratory_failure`, `is_kidney_failure`, `is_sepsis`, `is_diabetes`, `is_stroke_ich`, `is_cardiovascular`, `is_liver_disease`) serta etiologi mikroorganisme dan jenis spesimen.
- **Patient-Level Stratified Group Split (80:20)**:
  - Pembagian dilakukan di level **83 pasien unik** agar kuota mortalitas terdistribusi proporsional:
    - **Data Latih (80%)**: 66 pasien unik (54 Sembuh, 12 Meninggal) $\rightarrow$ 192 baris isolat.
    - **Data Uji (20%)**: 17 pasien unik (13 Sembuh, **4 Meninggal**) $\rightarrow$ 31 baris isolat.
  - Mengeliminasi efek pseudoreplikasi di mana sebelumnya data uji hanya memuat 1 pasien meninggal dengan 10 kultur berulang.
- **Penanganan Ketimpangan Kelas**:
  - Menerapkan **Cluster-Balanced & Cost-Sensitive Weighting** pada fungsi loss ($w_i = 1 / N_{\text{isolat}}$). Bobot kelas diseimbangkan 50:50 tanpa merusak kovarians data asli menggunakan resampling sintetis.
- **Validasi Silang (Repeated Patient-Level CV)**:
  - Dievaluasi menggunakan **5-Repeated 5-Fold Stratified Group CV (25 iterasi)** di mana 1 pasien unik dihitung tepat 1 suara.
  - **Model Terpilih**: **Gradient Boosting Classifier**
    - *Patient ROC-AUC (CV)*: **$0.9520 \pm 0.0182$** (95% CI: $[0.899 - 0.998]$)
    - *Patient PR-AUC (CV)*: **$0.8352 \pm 0.0544$**
    - *Patient Brier Score*: **$0.0713 \pm 0.0175$** (Kalibrasi probabilitas terbaik)
    - *Analisis Sensitivitas Homogen (68 pasien 7-digit murni)*: ROC-AUC **0.9462**.
- **Evaluasi Mandiri Data Uji (Level Pasien Unik, $n=17$)**:
  - Ambang batas klinis dioptimasi bebas bocor dari probabilitas OOF data latih (**$0.4811$**).
  - **Akurasi**: $82.4\%$
  - **Sensitivitas (Recall Pasien Kritis)**: **$75.0\%$** (3 dari 4 pasien meninggal terdeteksi)
  - **Spesifisitas**: **$84.6\%$** (11 dari 13 pasien sembuh bebas alarm palsu)
  - **Presisi**: **$60.0\%$** (3 benar dari 5 alarm risiko tinggi)
  - **ROC-AUC Pasien**: **$0.808$** (Evaluasi jujur, titik OOF bersesuaian dengan kurva ROC)

---

## 🚀 Panduan Penggunaan

### Prasyarat Dependensi
Pastikan Python 3.10+ terpasang, lalu instal paket dependensi:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib openpyxl ipywidgets
```

### Menjalankan Notebook
1. Buka Jupyter Lab atau VS Code pada folder repositori.
2. Buka `MBKM_2_Ivan_Nur_Azmi.ipynb`.
3. Jalankan sel berurutan (*Run All*).
4. Untuk inferensi langsung pasien individual, gunakan:
   - **Bagian 10 (Versi 1)**: Simulasi otomatis berbasis baris rekam medis riil data uji.
   - **Bagian 10 (Versi 2)**: Form interaktif GUI Clinical Decision Support System (CDSS).

### Inferensi Model Produksi (.pkl)
Artefak pipeline mandiri dapat langsung digunakan tanpa dependensi eksternal notebook:
```python
import joblib
import pandas as pd

bundle = joblib.load('models/model_luaran_rekam_medis.pkl')
pipeline = bundle['pipeline']
threshold = bundle['threshold_klinis']

# Contoh data baru
data_pasien = pd.DataFrame([{
    'Diagnosis': 'Pneumonia aspirasi + Syok septik',
    'Komorbid': 'Diabetes Melitus tipe 2',
    'Jenis sampel': 'sputum',
    'Nama bakteri': 'KLEBSIELLA PNEUMONIAE'
}])

prob_meninggal = pipeline.predict_proba(data_pasien)[0, 1]
keputusan = "RISIKO TINGGI (Meninggal)" if prob_meninggal >= threshold else "Risiko Rendah (Sembuh)"
print(f"Estimasi Risiko: {prob_meninggal:.1%} | Keputusan: {keputusan}")
```
