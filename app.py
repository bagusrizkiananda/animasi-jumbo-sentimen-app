import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv('data_preprocessed.csv')

# Tambahkan label dummy berbasis keyword sederhana
def simple_sentiment(text):
    positive_keywords = ['bagus', 'menarik', 'keren', 'lucu', 'hebat', 'baik']
    negative_keywords = ['jelek', 'buruk', 'bosan', 'gagal', 'kurang', 'tidak']
    text = text.lower()
    if any(word in text for word in positive_keywords):
        return 'positif'
    elif any(word in text for word in negative_keywords):
        return 'negatif'
    else:
        return 'netral'

df['label'] = df['normalized_text'].astype(str).apply(simple_sentiment)

st.title("Klasifikasi Sentimen Film Animasi Jumbo 🎬")
st.write("Model Naïve Bayes berbasis keyword untuk klasifikasi sentimen")

# Tampilkan data
st.subheader("Contoh Data")
st.write(df[['normalized_text', 'label']].head())

# Visualisasi distribusi sentimen
st.subheader("Distribusi Sentimen")
sentiment_counts = df['label'].value_counts()
st.bar_chart(sentiment_counts)

# Wordcloud untuk tiap kelas
st.subheader("Wordcloud per Sentimen")
for sentiment in ['positif', 'negatif']:
    st.markdown(f"**Sentimen {sentiment.capitalize()}**")
    text = " ".join(df[df['label'] == sentiment]['text_without_stopwords'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Train model dan prediksi
st.subheader("Prediksi Sentimen dari Input Pengguna")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['normalized_text'].astype(str))
y = df['label']

model = MultinomialNB()
model.fit(X, y)

user_input = st.text_area("Masukkan ulasan tentang film Animasi Jumbo:")
if st.button("Prediksi"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.write(f"**Hasil Prediksi Sentimen:** {prediction}")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")

st.caption("Model ini menggunakan label dummy berbasis keyword sederhana. Untuk hasil lebih akurat, gunakan data berlabel asli.")