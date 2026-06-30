"""
Attention mechanisms for the decoder.

Bahdanau (additive) attention — projects decoder hidden state
and encoder outputs into a common space, scores, and soft-weights.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense


class BahdanauAttention(tf.keras.layers.Layer):
    """Bahdanau-style additive attention."""

    def __init__(self, units: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        """
        Args:
            features: (batch, 49, 2048) — encoder feature map
            hidden:   (batch, units)    — decoder previous state
        Returns:
            context_vector, attention_weights
        """
        hidden_with_time = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context = attention_weights * features
        context = tf.reduce_sum(context, axis=1)
        return context, attention_weights
