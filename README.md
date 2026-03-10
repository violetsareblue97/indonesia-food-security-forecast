# Prediksi Kebutuhan Pangan Nasional 2024–2026

Proyek Machine Learning yang membangun sistem prediksi konsumsi pangan Indonesia menggunakan model **PatchTST (Patch Time Series Transformer)**. Data diambil dari Badan Pangan Nasional, mencakup konsumsi per provinsi dan komoditas dari tahun 2021 hingga 2023.

---

## Latar Belakang

Pengelolaan stok pangan nasional selama ini bersifat reaktif — pemerintah baru bertindak setelah masalah terjadi. Proyek ini mencoba membalik pola tersebut dengan memprediksi kebutuhan pangan untuk tiga tahun ke depan (2024, 2025, 2026), sehingga distribusi bisa direncanakan lebih awal.

---

## Dataset

- **Sumber**: [Badan Pangan Nasional — Rata-rata Konsumsi Kab/Kota](https://data.badanpangan.go.id/datasetpublications/rbe/rata-rata-konsumsi-kab-kota)
- **Periode**: 2021–2023
- **Cakupan**: 34 provinsi, berbagai komoditas pangan
- **Target**: Konsumsi pangan dalam kg/kapita/tahun

---

## Alur Kerja

**1. Pembersihan Data**
Memuat CSV dari API Badan Pangan, menangani nilai kosong dan duplikat, menyeragamkan nama kolom, dan mengagregasi data per kombinasi Provinsi–Komoditas–Tahun.

**2. EDA**
Melihat distribusi data, menghapus nilai negatif, memvisualisasikan tren konsumsi tahunan, dan mengidentifikasi komoditas dengan konsumsi tertinggi.

**3. Normalisasi dan Format**
Normalisasi target dengan `StandardScaler`. Membuat kolom `unique_id` (Provinsi_Komoditas) dan `ds` (datetime) sesuai format NeuralForecast.

**4. Pelatihan PatchTST**
```python
model = PatchTST(h=3, input_size=1, max_steps=50, scaler_type='standard', revin=True)
nf = NeuralForecast(models=[model], freq='YE')
nf.fit(df=df_train)
```

**5. Prediksi**
```python
forecasts = nf.predict()
hasil_2026 = forecasts[forecasts['ds'].dt.year == 2026]
```

---

## Performa Model

| Metrik | Nilai |
|---|---|
| R2 Score | 87.34% |
| MAE (normalized) | 0.0458 |
| Series terlatih | 120 kombinasi |

Catatan: nilai R2 dan MAE dihitung secara in-sample, bukan dari data uji terpisah. Prediksi 2024 paling bisa diandalkan, sedangkan prediksi 2026 lebih bersifat indikatif.

---

## Cara Menjalankan

**Install dependensi**
```bash
pip install -r requirements.txt
```

**Jalankan notebook**

Buka `Tugas_Akhir_Machine_Learning_Kelompok3.ipynb` di Google Colab atau Jupyter. Disarankan menggunakan Google Colab karena proses training membutuhkan waktu beberapa menit di CPU.

---

## Struktur File

```
├── Tugas_Akhir_Machine_Learning_Kelompok3.ipynb   # notebook utama
├── index.html                                      # website portofolio
├── requirements.txt
└── README.md
```

---

## Tech Stack

- Python, Pandas, NumPy
- NeuralForecast (PatchTST), PyTorch Lightning
- Scikit-learn (StandardScaler)
- Matplotlib, Seaborn

## Referensi
Badan Pangan Nasional. *Rata-rata Konsumsi Pangan Kab/Kota*. data.badanpangan.go.id
