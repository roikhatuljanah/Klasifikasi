import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from streamlit_option_menu import option_menu

# Load model dan objek yang dibutuhkan
scaler = joblib.load('model/scaler.pkl')
le_sex = joblib.load('model/label_encoder_Sex.pkl')
le_embarked = joblib.load('model/label_encoder_Embarked.pkl')
best_model = joblib.load('model/best_model.pkl')
features = joblib.load('model/selected_features.pkl')

# Set konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Keselamatan Penumpang Titanic",
    layout="centered",
    initial_sidebar_state="auto"
)

# Sidebar Menu
with st.sidebar:
    selected = option_menu("Menu Utama", ["Dashboard", "Visualisasi Data", "Perhitungan"],
                           icons=['house', 'bar-chart', 'calculator'], default_index=0)

# Fungsi encoding aman
def safe_encode(encoder, value, unknown_code=-1):
    classes = list(encoder.classes_)
    if value in classes:
        return encoder.transform([value])[0]
    else:
        return unknown_code

# Halaman Dashboard
if selected == "Dashboard":
    st.title("🚢 Prediksi Keselamatan Penumpang Titanic")
    st.write(
        "Aplikasi ini memprediksi apakah seorang penumpang Titanic akan **selamat** atau **tidak selamat** "
        "berdasarkan data pribadi mereka. Model terbaik yang digunakan adalah **Random Forest Classifier** "
        "yang telah dilatih dengan data historis Titanic."
    )

# Halaman Visualisasi
elif selected == "Visualisasi Data":
    st.title("📊 Visualisasi Fitur Terpenting")
    # Load ulang model jika diperlukan
    importances = best_model.feature_importances_
    feature_names = features
    importance_df = pd.DataFrame({'Fitur': feature_names, 'Pengaruh': importances})
    top_3_features = importance_df.sort_values(by='Pengaruh', ascending=False).head(3)

    st.subheader("Top 3 Fitur Paling Berpengaruh")
    st.dataframe(top_3_features)

    # Visualisasi
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top_3_features, x='Pengaruh', y='Fitur', palette='viridis')
    plt.title('Top 3 Fitur Penentu Keselamatan')
    st.pyplot(plt)

# Halaman Perhitungan
elif selected == "Perhitungan":
    st.title("🧮 Prediksi Keselamatan Penumpang")

    # Input user
    pclass = st.selectbox("Kelas Penumpang (Pclass)", [1, 2, 3], index=2)
    sex = st.selectbox("Jenis Kelamin", ['male', 'female'], index=1)
    age = st.slider("Umur", 0, 100, 25)
    sibsp = st.number_input("Jumlah Saudara/Istri/Suami", min_value=0, max_value=10, value=0)
    parch = st.number_input("Jumlah Orang Tua/Anak", min_value=0, max_value=10, value=0)
    fare = st.slider("Tarif (Fare)", 0.0, 600.0, 7.25)
    embarked = st.selectbox("Pelabuhan Keberangkatan (Embarked)", ['C', 'Q', 'S'], index=2)

    if st.button("Prediksi Keselamatan"):
        input_data = {
            'Pclass': pclass,
            'Sex': safe_encode(le_sex, sex),
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': safe_encode(le_embarked, embarked)
        }

        input_array = np.array([[input_data[feat] for feat in features]])
        scaled_input = scaler.transform(input_array)
        prediction = best_model.predict(scaled_input)[0]
        probability = best_model.predict_proba(scaled_input)[0][prediction]

        status = "💚 Selamat" if prediction == 1 else "💀 Tidak Selamat"
        st.success(f"**Prediksi:** {status}")
        st.info(f"Tingkat keyakinan model: {probability*100:.2f}%")