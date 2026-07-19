"""
Page 4 — Dokumentasi & Interpretasi
Project background, success metrics, and usage guide.
"""

import streamlit as st

st.title("Dokumentasi Proyek")

st.header("Latar Belakang")
st.markdown("""
**Image Captioning** adalah tugas yang menghubungkan Computer Vision dan Natural Language Processing,
di mana sebuah model harus mampu memahami isi gambar secara visual kemudian menghasilkan deskripsi
tekstual yang alami dan bermakna.

Proyek ini dibuat sebagai tugas akhir mata kuliah **Pembelajaran Mesin**.
""")

st.header("Dataset")
st.markdown("""
- **Flickr8k** — 8.091 gambar dengan 5 caption per gambar.
- Setiap caption mendeskripsikan aktivitas atau objek dalam gambar.
- Domain: aktivitas outdoor sehari-hari (manusia, hewan, pemandangan).
""")

st.header("Arsitektur Model")
st.markdown("""
| Komponen | Detail |
|----------|--------|
| Encoder | ResNet50 (pre-trained ImageNet, frozen) → 2048-d vector |
| Decoder LSTM | LSTM(512) + initial state dari proyeksi fitur |
| Decoder GRU | GRU(512) + initial state tunggal dari proyeksi fitur |
| Embedding | 5000 vocab → 256 dimensi |
| Total Param LSTM | 6.207.624 |
| Total Param GRU | 5.683.848 (8,4% lebih ringan) |
""")

st.header("Metrik Kesuksesan")
st.markdown("""
| Metrik | Fungsi |
|--------|--------|
| **Loss (Sparse Categorical Crossentropy)** | Mengukur kesalahan prediksi token per step |
| **Accuracy** | Persentase token yang diprediksi tepat |
| **BLEU-1 s.d. BLEU-4** | Kesamaan n-gram dengan referensi manusia |
| **SHAP Feature Importance** | Interpretasi fitur ResNet50 via XGBoost baseline |
""")

st.header("Cara Penggunaan")
st.markdown("""
1. **Notebook Pipeline:** Jalankan notebook 01-08 secara berurutan.
2. **Streamlit App:** `streamlit run app/app.py` untuk dashboard interaktif.
3. **Hyperparameter Tuning:** Terintegrasi di notebook 04 & 05 via KerasTuner.
4. **Interpretasi Model:** Notebook 08 untuk XGBoost + SHAP analysis.
""")

st.header("Struktur Proyek")
st.markdown("""
```
ml-image-captioning/
├── notebooks/         # 8 notebook Jupyter (01-08)
├── app/               # Streamlit multi-page dashboard
│   ├── app.py         # Entry point (navigasi)
│   ├── utils.py       # Fungsi bersama
│   └── pages/         # 4 halaman dashboard
├── src/               # Modul Python
├── model/             # Features & tokenizer
└── outputs/           # Hasil training & evaluasi
```
""")
