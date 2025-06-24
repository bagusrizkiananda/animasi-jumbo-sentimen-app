import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv('data_preprocessed.csv')

st.title("Analisis Sentimen Film Animasi Jumbo 🐘🎬")
st.write("Model Naïve Bayes berbasis Text Mining terhadap ulasan penonton")

# Tampilkan contoh data
st.subheader("Contoh Data")
st.write(df.head())

# Wordcloud
st.subheader("Wordcloud Semua Ulasan")
if 'clean_text' in df.columns:
    all_text = " ".join(df['clean_text'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
else:
    st.warning("Kolom 'clean_text' tidak ditemukan di dataset.")

# Training model jika ada label
st.subheader("Training Model Naïve Bayes (Opsional)")
label_col = st.selectbox("Pilih kolom label (jika tersedia):", ["(Tidak ada)"] + df.columns.tolist())
if label_col != "(Tidak ada)":
    try:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['clean_text'].astype(str))
        y = df[label_col]

        model = MultinomialNB()
        model.fit(X, y)
        st.success(f"Model dilatih menggunakan kolom label: {label_col}")

        # Prediksi input pengguna
        st.subheader("Prediksi Sentimen Ulasan")
        user_input = st.text_area("Masukkan ulasan tentang film Animasi Jumbo:")
        if st.button("Prediksi"):
            if user_input.strip() != "":
                input_vec = vectorizer.transform([user_input])
                prediction = model.predict(input_vec)[0]
                st.write(f"**Hasil Prediksi Sentimen:** {prediction}")
            else:
                st.warning("Silakan masukkan teks terlebih dahulu.")
    except Exception as e:
        st.error(f"Terjadi error saat melatih model: {e}")
else:
    st.info("Model tidak dilatih karena tidak ada kolom label yang dipilih.")

# Sumber Data
st.markdown("---")
st.caption("Dataset: Ulasan Film Animasi Jumbo (data_preprocessed.csv)")