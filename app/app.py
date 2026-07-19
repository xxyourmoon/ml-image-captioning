"""
Streamlit Multi-Page Application — Image Captioning Dashboard.

Usage:
    streamlit run app/app.py

Pages:
    1. EDA Dashboard   — Exploratory data analysis with Plotly
    2. Evaluasi Model   — BLEU score comparison + SHAP summary
    3. Demo Model       — Upload image → caption generation
    4. Dokumentasi      — Project documentation & usage guide
"""

import streamlit as st

st.set_page_config(
    page_title="Image Captioning — UAS Pembelajaran Mesin",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
)

pg = st.navigation([
    st.Page("views/01_eda_dashboard.py", title="Dashboard EDA"),
    st.Page("views/02_evaluasi.py",       title="Evaluasi Model"),
    st.Page("views/03_demo_model.py",     title="Demo Model"),
    st.Page("views/04_dokumentasi.py",    title="Dokumentasi"),
])

pg.run()
