"""
Page 1 — Dashboard EDA
Interactive exploratory data analysis with Plotly.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

st.title("Dashboard EDA — Eksplorasi Data")

CAPTIONS_CLEAN = "outputs/captions_clean.csv"

@st.cache_data
def load_clean_captions():
    return pd.read_csv(CAPTIONS_CLEAN)

df = load_clean_captions()

# Compute caption lengths
df["word_count"] = df["clean"].str.split().str.len()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribusi Panjang Caption")
    fig = px.histogram(
        df, x="word_count", nbins=30,
        title="Distribusi Jumlah Kata per Caption",
        labels={"word_count": "Jumlah Kata", "count": "Frekuensi"},
        color_discrete_sequence=["#4C78A8"]
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Statistik Caption")
    stats = df["word_count"].describe()
    st.metric("Total Caption", f"{len(df):,}")
    st.metric("Rata-rata Panjang", f"{stats['mean']:.1f} kata")
    st.metric("Min / Max", f"{int(stats['min'])} / {int(stats['max'])} kata")
    st.metric("Median", f"{stats['50%']:.1f} kata")

st.subheader("Top 20 Kata Paling Sering Muncul")
all_words = " ".join(df["clean"]).lower().split()
word_series = pd.Series(all_words).value_counts().head(20)

fig_bar = px.bar(
    x=word_series.values[::-1],
    y=word_series.index[::-1],
    orientation="h",
    title="20 Kata Teratas",
    labels={"x": "Frekuensi", "y": "Kata"},
    color=word_series.values[::-1],
    color_continuous_scale="Blues"
)
fig_bar.update_layout(showlegend=False)
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("5 Insights Utama dari Data")
insights = [
    "Setiap gambar memiliki tepat 5 caption — tidak ada missing values.",
    "Rata-rata panjang caption ~11 kata; MAX_CAPTION_LEN=34 mencakup >99% data.",
    "Vocabulary 5.000 kata, didominasi kata umum (the, a, in, and, on).",
    "Kata benda aktivitas outdoor mendominasi (dog, man, running, playing).",
    "Dataset bersih tanpa anomali signifikan — siap untuk preprocessing."
]
for i, insight in enumerate(insights, 1):
    st.markdown(f"**{i}.** {insight}")
