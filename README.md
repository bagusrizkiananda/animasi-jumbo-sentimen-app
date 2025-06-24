# Analisis Sentimen Film Animasi Jumbo 🎬🐘

Aplikasi Streamlit untuk analisis sentimen ulasan penonton terhadap film **Animasi Jumbo**, menggunakan metode **Naïve Bayes** dan teknik **text mining**.

## 📂 Fitur Aplikasi
- Visualisasi distribusi sentimen (positif, negatif, netral)
- Wordcloud dari seluruh ulasan
- Prediksi sentimen berdasarkan input teks pengguna
- Model dilatih langsung tanpa file `.pkl`

## 📁 Struktur Proyek
```
├── app.py                 # Streamlit App utama
├── data_preprocessed.csv  # Dataset ulasan film Animasi Jumbo
├── requirements.txt       # Dependencies untuk Streamlit Cloud
└── README.md              # Dokumentasi ini
```

## 🚀 Cara Menjalankan
### 1. Secara Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Melalui Streamlit Cloud
- Upload semua file ini ke GitHub
- Masuk ke [streamlit.io/cloud](https://streamlit.io/cloud)
- Hubungkan ke repo GitHub kamu
- Pilih `app.py` sebagai entry point
- Jalankan 🎉

## 🧠 Teknologi
- Python
- Streamlit
- Scikit-learn
- TfidfVectorizer
- Naïve Bayes
- WordCloud

## 📄 Dataset
Dataset berasal dari ulasan penonton terhadap film **Animasi Jumbo** yang telah dipreprocessing, termasuk stopword removal dan pembersihan teks.