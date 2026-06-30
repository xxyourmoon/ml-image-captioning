"""
CNN + LSTM + Attention — primary architecture.

Builds the encoder-decoder graph and exposes train / predict helpers.
"""

from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Embedding,
                                     Input, Add)

from src.models.attention import BahdanauAttention
from src.config import (VOCAB_SIZE, MAX_CAPTION_LEN, EMBEDDING_DIM,
                        LSTM_UNITS, DROPOUT_RATE)


def build_lstm_model(feature_dim: int = 2048) -> Model:
    """
    Build CNN-LSTM-Attention model.

    Args:
        feature_dim: size of the image feature vector (ResNet50 → 2048)

    Returns:
        compiled tf.keras.Model
    """
    # --- Image input ---
    img_input = Input(shape=(feature_dim,), name="image_input")
    img_dense = Dense(EMBEDDING_DIM, activation="relu")(img_input)
    img_drop  = Dropout(DROPOUT_RATE)(img_dense)

    # --- Caption input ---
    cap_input = Input(shape=(MAX_CAPTION_LEN,), name="caption_input")
    cap_embed = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(cap_input)

    # --- LSTM Decoder ---
    lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
    lstm_out, state_h, state_c = lstm(cap_embed)

    # --- Attention ---
    attention = BahdanauAttention(LSTM_UNITS)
    context, _ = attention(img_drop, state_h)

    # --- Decode ---
    decoder_input = tf.concat([context, state_h], axis=-1)
    decoder_dense = Dense(VOCAB_SIZE, activation="softmax", name="output")
    output = decoder_dense(decoder_input)

    model = Model(inputs=[img_input, cap_input], outputs=output, name="lstm_attention")
    return model


def load_lstm_weights(path: str) -> Model:
    """Load a saved LSTM model weight file."""
    model = build_lstm_model()
    model.load_weights(path)
    return model
