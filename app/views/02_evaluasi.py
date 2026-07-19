"""
Page 2 — Evaluasi Model
BLEU score comparison table, bar chart, and SHAP summary plot.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

st.title("Evaluasi Model — LSTM vs GRU")

BLEU_PATH = "outputs/evaluation/bleu_scores.csv"
SHAP_PATH = "outputs/evaluation/shap_summary.png"

@st.cache_data
def load_bleu():
    return pd.read_csv(BLEU_PATH)

bleu_df = load_bleu()

st.subheader("Tabel BLEU Score")
bleu_display = bleu_df.copy()
bleu_display.columns = ["Metric", "LSTM", "GRU", "Delta (GRU - LSTM)"]
bleu_display["Metric"] = bleu_display["Metric"].str.replace("bleu_", "BLEU-")
st.dataframe(bleu_display, use_container_width=True, hide_index=True)

st.subheader("Perbandingan BLEU Score")
fig = px.bar(
    bleu_df,
    x="metric",
    y=["lstm", "gru"],
    barmode="group",
    title="Perbandingan BLEU Score: LSTM vs GRU",
    labels={"value": "BLEU Score", "metric": "Metric",
            "variable": "Model"},
    color_discrete_map={"lstm": "#4C78A8", "gru": "#54A24B"}
)
fig.update_layout(legend_title_text="Model")
fig.update_xaxes(ticktext=["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"],
                 tickvals=["bleu_1", "bleu_2", "bleu_3", "bleu_4"])
st.plotly_chart(fig, use_container_width=True)

st.subheader("SHAP Summary Plot — Interpretasi Fitur ResNet50")
if os.path.exists(SHAP_PATH):
    shap_img = Image.open(SHAP_PATH)
    st.image(shap_img, caption="SHAP summary plot: fitur ResNet50 yang paling berkontribusi",
             use_column_width=True)
    st.markdown(
        "**Interpretasi:** SHAP menunjukkan fitur (dimensi) ResNet50 mana yang "
        "paling berkontribusi terhadap prediksi objek. Warna merah = nilai fitur "
        "tinggi, warna biru = nilai fitur rendah."
    )
else:
    st.warning("SHAP summary plot belum tersedia. "
               "Jalankan notebook 08 terlebih dahulu.")

st.subheader("Kesimpulan Evaluasi")
st.markdown("""
- GRU **konsisten mengungguli LSTM** pada seluruh metrik BLEU (BLEU-1 hingga BLEU-4).
- Selisih performa kecil (~0.005-0.009), tetapi konsisten.
- GRU memiliki **~8,4% lebih sedikit parameter** dan **~12% lebih cepat** dalam training.
- Untuk dataset Flickr8k, GRU yang lebih sederhana sudah mencukupi.
""")
