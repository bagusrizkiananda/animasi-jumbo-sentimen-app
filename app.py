import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv("data_preprocessed.csv")
df = df[['normalized_text']].dropna().reset_index(drop=True)

st.title("Klasifikasi Sentimen Manual: Animasi Jumbo 🎬")
st.write("Silakan pilih sentimen (positif atau negatif) untuk setiap kalimat berikut:")

# Batas jumlah yang ditampilkan
num_rows = st.slider("Jumlah data yang ditampilkan:", 1, min(20, len(df)), 5)

# Tempat simpan jawaban user
user_labels = []

for i in range(num_rows):
    st.markdown(f"**{i+1}.** {df.loc[i, 'normalized_text']}")
    label = st.radio(f"Pilih sentimen untuk kalimat {i+1}:", ['positif', 'negatif'], key=i)
    user_labels.append(label)

if st.button("Simpan Jawaban"):
    labeled_df = df.iloc[:num_rows].copy()
    labeled_df['label_user'] = user_labels
    labeled_df.to_csv("hasil_label_user.csv", index=False)
    st.success("Label berhasil disimpan ke `hasil_label_user.csv`!")

st.caption("Aplikasi ini digunakan untuk pelabelan manual dua kelas: positif dan negatif.")
