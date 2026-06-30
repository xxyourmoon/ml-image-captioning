"""
Streamlit application — Image Captioning Demo.

Usage:
    streamlit run app/app.py

Pipeline:
    Upload image → ResNet50 extract → LSTM/GRU decode → Show caption
"""

import streamlit as st

st.set_page_config(page_title="Image Captioning", layout="centered")
st.title("Image Captioning — CNN + LSTM / GRU")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Input image", use_container_width=True)
    st.info("Caption generation will be implemented in the next phase.")
