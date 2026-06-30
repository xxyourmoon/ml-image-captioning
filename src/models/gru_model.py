"""
CNN + GRU — baseline architecture (no attention).

Simpler, faster-to-train counterpart of the primary LSTM model.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Dense, Dropout, Embedding,
                                     GRU, Input)

from src.config import (VOCAB_SIZE, MAX_CAPTION_LEN, EMBEDDING_DIM,
                        GRU_UNITS, DROPOUT_RATE)


def build_gru_model(feature_dim: int = 2048) -> Model:
    """
    Build CNN-GRU model (baseline, no attention).

    Args:
        feature_dim: size of the image feature vector

    Returns:
        compiled tf.keras.Model
    """
    img_input = Input(shape=(feature_dim,), name="image_input")
    img_dense = Dense(EMBEDDING_DIM, activation="relu")(img_input)
    img_drop  = Dropout(DROPOUT_RATE)(img_dense)

    cap_input = Input(shape=(MAX_CAPTION_LEN,), name="caption_input")
    cap_embed = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(cap_input)

    gru = GRU(GRU_UNITS, return_sequences=False)
    gru_out = gru(cap_embed)

    combined = tf.concat([img_drop, gru_out], axis=-1)
    output = Dense(VOCAB_SIZE, activation="softmax", name="output")(combined)

    model = Model(inputs=[img_input, cap_input], outputs=output, name="gru_baseline")
    return model


def load_gru_weights(path: str) -> Model:
    """Load a saved GRU model weight file."""
    model = build_gru_model()
    model.load_weights(path)
    return model
