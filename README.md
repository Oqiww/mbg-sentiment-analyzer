---
title: MBG Sentiment Analyzer
emoji: 🍱
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.49.1
app_file: app.py
pinned: false
license: mit
---

# 🍱 MBG Sentiment Analyzer (IndoBERT-base-p1)

Aplikasi berbasis web (Streamlit) kelas produksi untuk **analisis sentimen opini publik terhadap program Makan Bergizi Gratis (MBG)** di Indonesia. Sistem ini ditenagai oleh model deep learning **IndoBERT-base-p1** yang telah di-finetune untuk mengklasifikasikan sentimen ke dalam 3 kelas: **Positif**, **Netral**, dan **Negatif**.

---

## 📌 Ringkasan Proyek

Program **Makan Bergizi Gratis (MBG)** merupakan salah satu kebijakan nasional dengan volume diskursus publik yang sangat masif di berbagai platform media sosial. Komentar masyarakat mencakup berbagai aspek seperti kualitas makanan, transparansi anggaran, pemerataan distribusi, hingga kritik dan harapan.

Aplikasi ini dirancang untuk:
1. **Analisis Real-Time**: Menganalisis sentimen dari satu kalimat atau komentar dengan visualisasi confidence score dan distribusi probabilitas softmax.
2. **Preprocessing Transparency**: Menampilkan tahapan preprocessing teks (*Stream B*) secara interaktif agar pengguna dapat memverifikasi bagaimana emoji, slang, mention, dan karakter berulang dinormalisasi.
3. **Batch Processing**: Memungkinkan analisis ratusan hingga ribuan komentar sekaligus melalui upload file CSV, lengkap dengan agregasi statistik dan fitur ekspor CSV.
4. **Desain UI/UX Modern & Responsif**: Tampilan antarmuka yang elegan, informatif, dan mudah diakses oleh pengambil kebijakan, analis data, maupun masyarakat umum.

---

## 🧠 Arsitektur Model & Pipeline

### 1. Model Spesifikasi
- **Base Architecture**: `indobenchmark/indobert-base-p1` (Transformer Encoder, 12 layers, 768 hidden size, 12 attention heads)
- **Task Head**: `BertForSequenceClassification` (3 Output Classes)
- **Tokenization**: Fast WordPiece Tokenizer (`tokenizer.json` / `AutoTokenizer`), `max_length: 256`, `padding: max_length`, `truncation: True`
- **Class Mapping**:
  - `0`: **Positif** (dukungan, apresiasi, harapan baik, kepuasan)
  - `1`: **Netral** (berita faktual, pengumuman jadwal, pernyataan seimbang tanpa emosi)
  - `2`: **Negatif** (keluhan makanan basi, kekhawatiran korupsi anggaran, kritik tajam)

### 2. Pipeline Preprocessing (*Stream B*)
Preprocessing pipeline direplikasi secara identik dengan pipeline saat pelatihan model:
1. **Unicode Normalization**: `unicodedata.normalize("NFKC", text)`
2. **Indonesian Emoji Semantic Mapping**: Mengonversi emoji ke semantic text token dalam Bahasa Indonesia (misal `👍` → `emoji_jempol_setuju`, `😡` → `emoji_marah`, `🗿` → `emoji_sarkas_batu`).
3. **Case Folding**: `text.lower()`
4. **Noise Removal**: Menghapus URLs (`http...`), `@mentions`, dan `#hashtags`.
5. **Curated Slang Normalization**: Kamus slang terkurasi khusus diskursus MBG (misal `bgt` → `banget`, `korup` → `korupsi`, `laukny` → `lauknya`).
6. **Character Repetition Normalization**: Menghilangkan pengulangan huruf lebih dari 2 kali (misal `enaaaak` → `enaak`).
7. **Whitespace Normalization**: Merapikan spasi berlebih dan strip whitespace.

---

## 📁 Struktur Direktori

```plaintext
week5_mbg/
├── app.py                          # Main Streamlit web application
├── requirements.txt                # Dependensi Python untuk runtime
├── README.md                       # Dokumentasi lengkap proyek
├── model/                          # Bobot model dan konfigurasi tokenizer
│   ├── config.json                 # HuggingFace model config
│   ├── model.safetensors           # Bobot IndoBERT fine-tuned (Safetensors format)
│   ├── tokenizer.json              # Fast tokenizer JSON
│   ├── tokenizer_config.json       # Konfigurasi tokenizer
│   └── slang_dict_mbg.json         # Kamus normalisasi kata gaul/slang
└── src/                            # Modul backend logika inferensi & preprocessing
    ├── __init__.py
    ├── preprocessing.py            # Implementasi Stream B preprocessing
    └── inference.py                # Model loader, tokenizer, dan predict pipeline
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Prasyarat Sistem
- Python 3.9 - 3.12
- GPU (CUDA) opsional tapi direkomendasikan; CPU sepenuhnya didukung secara otomatis.

### 2. Instalasi Dependensi

Buka terminal di direktori proyek dan jalankan:

```bash
pip install -r requirements.txt
```

### 3. Menjalankan Server Streamlit

Jalankan perintah berikut:

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada URL default:
`http://localhost:8501`

---

## 💡 Fitur Utama Aplikasi

### 1. Single Comment Analyzer
- Masukkan teks komentar masyarakat secara bebas (hingga 1.000 karakter).
- Tersedia tombol cepat untuk mencoba **Contoh Komentar**:
  - *Positif* (Apresiasi & kebermanfaatan)
  - *Netral* (Berita/Jadwal distribusi)
  - *Negatif* (Keluhan kualitas / anggaran)
  - *Ambigu / Sarkasme* (Sindiran lauk minim)
- Menampilkan:
  - Label sentimen utama dengan badge & ikon representatif.
  - Skor keyakinan (*confidence score*) persentase.
  - Progress bar distribusi ketiga probabilitas (Softmax).
  - Tab **"Detail Preprocessing (Stream B)"** yang memperlihatkan perubahan teks sebelum dan sesudah normalisasi.
  - Interpretasi naratif untuk memudahkan pemahaman pengguna non-teknis.

### 2. Batch Analysis (Upload CSV)
- Unggah file CSV yang memiliki kolom `comment`.
- Proses inferensi sekuensial dengan *real-time progress bar*.
- Kartu rekapitulasi distribusi sentimen total.
- Tabel hasil prediksi interaktif.
- Tombol **"Download Hasil CSV"** untuk mengekspor data hasil klasifikasi.

### 3. Halaman "Cara Kerja"
- Panduan transparan mengenai metodologi NLP yang digunakan.
- Penjelasan arsitektur Transformer dan keunggulan IndoBERT untuk konteks bahasa Indonesia sehari-hari.
- Disclaimer batasan model terkait sarkasme mendalam dan konteks implisit.

---

## ⚠️ Batasan & Catatan Etika
1. **Sarkasme & Bahasa Gaul Kriptik**: Walaupun model mengenali semantic token dan slang umum, sarkasme berlapis atau konteks lokal yang sangat spesifik dapat menyebabkan misklasifikasi.
2. **Transparansi**: Output model harus dijadikan alat bantu agregasi opini publik, bukan sebagai penilai absolut tanpa verifikasi manusia untuk keputusan berisiko tinggi.

---

## 👨‍💻 Kontributor
- **Model Training & Notebook**: Syauqi Gathan Setyapratama (Week 4 BD - LAS26)
- **Deployment & Productionization**: Tim ML & MLOps MBG
