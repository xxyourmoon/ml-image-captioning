"""
Page 3 — Demo Model
Upload image and generate caption (ported from original app.py).
"""

import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from utils import (
    load_model_and_tokenizer, extract_features,
    greedy_caption, beam_search_caption
)

st.title("Demo — Image Captioning")
st.markdown("Upload gambar untuk menghasilkan caption otomatis.")

col1, col2 = st.columns(2)
with col1:
    model_choice = st.radio("Model", ["LSTM", "GRU"], horizontal=True)
with col2:
    decoding = st.radio("Decoding", ["Greedy", "Beam-3"], horizontal=True)

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image_bytes = uploaded.read()
    st.image(image_bytes, caption="Input Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        try:
            t0 = time.time()

            model, index_word, start_id, end_id = load_model_and_tokenizer(
                model_choice
            )
            feat = extract_features(image_bytes)

            if decoding == "Greedy":
                cap = greedy_caption(model, feat, index_word, start_id, end_id)
            else:
                cap = beam_search_caption(model, feat, index_word,
                                          start_id, end_id, beam_size=3)

            elapsed = time.time() - t0

            st.success(cap)
            st.caption(f"Generated in {elapsed:.2f}s using "
                       f"{model_choice} + {decoding}")

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Upload an image to get started.")
