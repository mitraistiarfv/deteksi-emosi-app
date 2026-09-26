
import streamlit as st
import joblib
import re

# Fungsi pra-pemrosesan teks (sama seperti yang digunakan saat pelatihan)
def preprocess_text(text):
    text = text.lower() # Ubah teks menjadi huruf kecil
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Hapus karakter non-alfabet
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih
    return text

# Muat model, vectorizer, dan encoder
try:
    svm_model = joblib.load('svm_emotion_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    # st.success("Model, Vectorizer, dan Label Encoder berhasil dimuat!") # Streamlit UI will handle this
except Exception as e:
    st.error(f"Gagal memuat komponen model: {e}")
    st.stop() # Hentikan aplikasi jika gagal memuat komponen

# Judul Aplikasi
st.title("Aplikasi Deteksi Emosi Berbasis Teks")
st.write("Masukkan teks di bawah untuk memprediksi emosinya.")

# Input teks dari pengguna
user_input = st.text_area("Tulis sesuatu di sini:", "Saya sangat senang hari ini!")

if st.button("Deteksi Emosi"):
    if user_input:
        # Pra-pemrosesan teks input
        processed_input = preprocess_text(user_input)
        
        # Vektorisasi teks input
        vectorized_input = tfidf_vectorizer.transform([processed_input])
        
        # Prediksi emosi
        prediction_encoded = svm_model.predict(vectorized_input)
        
        # Dekode hasil prediksi ke label asli
        predicted_emotion = label_encoder.inverse_transform(prediction_encoded)[0]
        
        st.success(f"Emosi yang terdeteksi: **{predicted_emotion.capitalize()}**")
    else:
        st.warning("Mohon masukkan teks untuk deteksi emosi.")
