import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv("data_preprocessed.csv")

# Buat label dummy berdasarkan keyword
def simple_sentiment(text):
    positive_keywords = ['bagus', 'menarik', 'keren', 'lucu', 'hebat', 'baik']
    negative_keywords = ['jelek', 'buruk', 'bosan', 'gagal', 'kurang', 'tidak']
    text = text.lower()
    if any(word in text for word in positive_keywords):
        return 'positif'
    elif any(word in text for word in negative_keywords):
        return 'negatif'
    else:
        return None

df['label'] = df['normalized_text'].astype(str).apply(simple_sentiment)
df = df[df['label'].isin(['positif', 'negatif'])]

# UI Streamlit
st.title("Filter Komentar berdasarkan Sentimen 🎭")
sentimen = st.selectbox("Pilih jenis sentimen yang ingin ditampilkan:", ['positif', 'negatif'])

filtered_df = df[df['label'] == sentimen]

st.write(f"Menampilkan komentar dengan sentimen **{sentimen}**:")
st.dataframe(filtered_df[['normalized_text']].reset_index(drop=True))
