import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide"
)

# =============================
# LOAD DATASET
# =============================
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# =============================
# SPLIT & NORMALISASI
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =============================
# SIDEBAR
# =============================
st.sidebar.title("⚙ Pengaturan Model")

k = st.sidebar.slider(
    "Jumlah Tetangga (K)",
    min_value=1,
    max_value=15,
    value=5,
    step=2
)

# =============================
# TRAIN MODEL
# =============================
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# =============================
# EVALUASI MODEL
# =============================
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.subheader("📊 Evaluasi Model")
st.sidebar.write(f"Akurasi Model: **{accuracy*100:.2f}%**")

with st.sidebar.expander("Classification Report"):
    st.text(classification_report(y_test, y_pred))

with st.sidebar.expander("Confusion Matrix"):
    st.write(confusion_matrix(y_test, y_pred))

# =============================
# TITLE
# =============================
st.title("❤️ Prediksi Penyakit Jantung")
st.write("Aplikasi prediksi penyakit jantung berbasis **Machine Learning (KNN)** dengan analisis kemiripan data.")

st.warning("""
⚠ **Disclaimer**
Aplikasi ini hanya sebagai alat bantu prediksi berbasis Machine Learning  
dan **bukan pengganti diagnosis dokter**.
""")

# =============================
# MAPPING
# =============================
mapping_sex = {"Perempuan": 0, "Laki-Laki": 1}
mapping_cp = {
    "Angina Tipikal": 1,
    "Angina Atypikal": 2,
    "Nyeri Non-Angina": 3,
    "Asimptomatik": 4
}
mapping_fbs = {"< 120 mg/dl": 0, "≥ 120 mg/dl": 1}
mapping_restecg = {
    "Normal": 0,
    "Kelainan ST-T": 1,
    "Hipertrofi Ventrikel Kiri": 2
}
mapping_exang = {"Tidak": 0, "Ya": 1}
mapping_slope = {
    "Upsloping": 0,
    "Datar (Flat)": 1,
    "Downsloping": 2
}
mapping_thal = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}

# =============================
# FORM INPUT
# =============================
st.subheader("📝 Input Data Pasien")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur", 1, 120, 29)
    sex = st.selectbox("Jenis Kelamin", mapping_sex.keys())
    cp = st.selectbox("Tipe Nyeri Dada", mapping_cp.keys())
    trestbps = st.number_input("Tekanan Darah Istirahat (mmHg)", 80, 200, 120)
    chol = st.number_input("Kolesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Gula Darah Puasa", mapping_fbs.keys())

with col2:
    restecg = st.selectbox("Hasil EKG Istirahat", mapping_restecg.keys())
    thalach = st.number_input("Detak Jantung Maksimum", 60, 220, 150)
    exang = st.selectbox("Nyeri Dada Saat Olahraga", mapping_exang.keys())
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope ST Segment", mapping_slope.keys())
    ca = st.number_input("Jumlah Pembuluh Darah (0–3)", 0, 3, 0)
    thal = st.selectbox("Thalassemia", mapping_thal.keys())

# =============================
# PREDIKSI
# =============================
if st.button("🔍 Prediksi"):

    with st.spinner("Memproses prediksi..."):
        time.sleep(1)

    user_input = np.array([[
        age,
        mapping_sex[sex],
        mapping_cp[cp],
        trestbps,
        chol,
        mapping_fbs[fbs],
        mapping_restecg[restecg],
        thalach,
        mapping_exang[exang],
        oldpeak,
        mapping_slope[slope],
        ca,
        mapping_thal[thal]
    ]])

    user_scaled = scaler.transform(user_input)

    prediction = knn.predict(user_scaled)[0]
    probability = knn.predict_proba(user_scaled)[0][1] * 100

    # =============================
    # HASIL PREDIKSI
    # =============================
    st.subheader("📌 Hasil Prediksi")

    if prediction == 1:
        st.error(f"⚠ **Risiko Penyakit Jantung Terdeteksi** ({probability:.2f}%)")
        st.write("""
        Hasil ini menunjukkan adanya kemiripan dengan pasien yang memiliki
        penyakit jantung pada dataset.
        **Disarankan untuk berkonsultasi dengan dokter.**
        """)
    else:
        st.success(f"✔ **Tidak Terdeteksi Penyakit Jantung** ({probability:.2f}%)")
        st.write("""
        Data Anda lebih mirip dengan pasien tanpa penyakit jantung.
        Tetap jaga pola hidup sehat.
        """)

    # =============================
    # SIMILARITY
    # =============================
    st.subheader("🔗 Analisis Kemiripan Data")

    sim = cosine_similarity(user_scaled, X_scaled)[0]
    similarity_percent = max(0, np.max(sim) * 100)

    st.write(f"Tingkat kemiripan tertinggi: **{similarity_percent:.2f}%**")

    top_n = 5
    top_idx = np.argsort(sim)[-top_n:][::-1]

    st.write("📊 **5 Pasien Paling Mirip di Dataset**")
    st.dataframe(df.iloc[top_idx])

    # =============================
    # VISUALISASI DATA PASIEN
    # =============================
    st.subheader("📈 Visualisasi Data Pasien")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(X.columns, user_input[0])
    ax.set_xticklabels(X.columns, rotation=90)
    st.pyplot(fig)

    # =============================
    # SIMPAN RIWAYAT
    # =============================
    if st.button("💾 Simpan Hasil Prediksi"):
        result = pd.DataFrame(user_input, columns=X.columns)
        result["prediction"] = prediction
        result["probability"] = probability
        result.to_csv("riwayat_prediksi.csv", mode="a", header=False, index=False)
        st.success("Hasil prediksi berhasil disimpan!")
