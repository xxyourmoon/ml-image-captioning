"""
Shared utilities for the Streamlit multi-page app.
"""

import os
import pickle
import time

import numpy as np
import streamlit as st

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

LSTM_DIR = "outputs/training/lstm"
GRU_DIR  = "outputs/training/gru"
IMG_SIZE = (224, 224)
MAX_CAPTION_LEN = 34


@st.cache_resource
def load_resnet():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")


@st.cache_resource
def load_model_and_tokenizer(model_choice: str):
    model_dir = LSTM_DIR if model_choice == "LSTM" else GRU_DIR
    model_file = os.path.join(model_dir, "lstm_best.keras" if model_choice == "LSTM"
                              else "gru_best.keras")
    tok_file   = os.path.join(model_dir, "tokenizer.pkl")

    if not os.path.exists(model_file):
        st.error(f"Model file not found: `{model_file}`. "
                 f"Upload the model files to Hugging Face Spaces via the Files tab "
                 f"or run the upload script: "
                 f"`bash scripts/upload_model_to_spaces.sh YOUR_SPACE_NAME`")
        st.stop()
    if not os.path.exists(tok_file):
        st.error(f"Tokenizer file not found: `{tok_file}`")
        st.stop()

    model = tf.keras.models.load_model(model_file)
    with open(tok_file, "rb") as f:
        tok = pickle.load(f)

    word_index = tok.word_index
    index_word = {v: k for k, v in word_index.items()}
    start_id = word_index.get("<start>", 1)
    end_id   = word_index.get("<end>",   2)

    return model, index_word, start_id, end_id


def extract_features(image_bytes):
    cnn = load_resnet()
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    img = tf.expand_dims(img, 0)
    feat = cnn.predict(img, verbose=0)
    return feat[0].astype(np.float32)


def greedy_caption(model, image_feat, index_word, start_id, end_id,
                   max_len=MAX_CAPTION_LEN):
    seq = [start_id]
    img_feat = image_feat.astype(np.float32).reshape(1, -1)

    for step in range(max_len - 1):
        input_seq = pad_sequences([seq], maxlen=max_len - 1, padding="post",
                                  dtype=np.int32)
        preds = model.predict([img_feat, input_seq], verbose=0)
        next_id = int(np.argmax(preds[0, step]))
        if next_id == end_id:
            break
        seq.append(next_id)

    words = [index_word.get(t, "<oov>") for t in seq[1:]]
    return " ".join(words)


def beam_search_caption(model, image_feat, index_word, start_id, end_id,
                        beam_size=3, max_len=MAX_CAPTION_LEN):
    img_feat = image_feat.astype(np.float32).reshape(1, -1)
    seqs = [([start_id], 0.0)]

    for step in range(max_len - 1):
        all_candidates = []
        for seq, score in seqs:
            input_seq = pad_sequences([seq], maxlen=max_len - 1, padding="post",
                                      dtype=np.int32)
            preds = model.predict([img_feat, input_seq], verbose=0)
            probs = tf.nn.softmax(preds[0, step] / 1.0).numpy()
            top_indices = np.argsort(probs)[-beam_size:][::-1]

            for idx in top_indices:
                candidate_seq = seq + [int(idx)]
                candidate_score = score + np.log(probs[idx] + 1e-10)
                all_candidates.append((candidate_seq, candidate_score))

        seqs = sorted(all_candidates, key=lambda x: x[1],
                      reverse=True)[:beam_size]

        if all(s[-1] == end_id for s, _ in seqs):
            break

    best_seq = max(seqs, key=lambda x: x[1])[0]
    words = [index_word.get(t, "<oov>") for t in best_seq
             if t != start_id and t != end_id]
    return " ".join(words)
