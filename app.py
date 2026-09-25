
# Install Streamlit jika belum terinstal (hanya perlu dijalankan sekali)
# !pip install streamlit # Ini tidak diperlukan di file lokal jika sudah terinstal

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Pastikan resource NLTK sudah diunduh (untuk deployment, ini perlu dilakukan di lingkungan server)
nltk.download('stopwords', quiet=True)

# --- Fungsi Pra-pemrosesan Teks (Harus sama persis dengan saat pelatihan) ---
stop_words = set(stopwords.words('indonesian')) # Gunakan stop words bahasa Indonesia
stemmer = PorterStemmer() # Porter Stemmer untuk bahasa Inggris

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # escape backslash for the string literal
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# --- Pemuatan Model dan Vectorizer ---
# Sesuaikan path ini jika Anda menjalankan di lingkungan lokal/cloud yang berbeda
# Untuk deployment Streamlit Cloud, file ini sebaiknya diletakkan di root repository GitHub Anda
# Di sini, diasumsikan model dan vectorizer berada di direktori yang sama dengan app.py jika dijalankan lokal
model_path = 'svm_emotion_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

@st.cache_resource # Cache resource agar model tidak dimuat ulang setiap kali interaksi
def load_model_and_vectorizer():
    # Memuat dari lokasi relatif jika app.py dan .pkl ada di direktori yang sama
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# --- Antarmuka Pengguna Streamlit ---
st.title('Aplikasi Deteksi Emosi ISEAR')
st.write('Masukkan teks untuk memprediksi emosi.')

user_input = st.text_area('Teks Anda:', '')

if st.button('Prediksi Emosi'):
    if user_input:
        # Pra-pemrosesan teks input
        processed_input = preprocess_text(user_input)
        
        # Vektorisasi teks input
        X_input = vectorizer.transform([processed_input])
        
        # Prediksi emosi
        prediction = model.predict(X_input)
        
        st.success(f'Emosi yang Diprediksi: **{prediction[0]}**')
    else:
        st.warning('Mohon masukkan teks terlebih dahulu.')
