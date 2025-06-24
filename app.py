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

# Visualisasi awal
st.subheader("Distribusi Sentimen")
sentiment_counts = df['label'].value_counts()
st.bar_chart(sentiment_counts)

# Wordcloud
st.subheader("Wordcloud Semua Ulasan")
all_text = " ".join(df['clean_text'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(plt)

# Training model
st.subheader("Training Model Naïve Bayes")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'].astype(str))
y = df['label']

model = MultinomialNB()
model.fit(X, y)
st.success("Model telah dilatih!")

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

# Sumber Data
st.markdown("---")
st.caption("Dataset: Ulasan Film Animasi Jumbo (data_preprocessed.csv)")