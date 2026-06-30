"""
Tokenizer wrapper — builds vocabulary, fits, saves, and loads.

Used by notebooks/04_tokenizer.ipynb.
"""

import pickle
from pathlib import Path
from typing import List, Optional

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import VOCAB_SIZE, MAX_CAPTION_LEN, MODEL_DIR


def build_tokenizer(captions: List[str],
                    vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """Fit a TF Tokenizer on the caption corpus."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts(captions)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: Optional[Path] = None) -> Path:
    """Persist tokenizer to model/tokenizer.pkl."""
    if path is None:
        path = MODEL_DIR / "tokenizer.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)
    return path


def load_tokenizer(path: Optional[Path] = None) -> Tokenizer:
    """Load a previously saved tokenizer."""
    if path is None:
        path = MODEL_DIR / "tokenizer.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def caption_to_sequence(tokenizer: Tokenizer,
                        caption: str,
                        max_len: int = MAX_CAPTION_LEN):
    """Convert a single caption string to a padded integer sequence."""
    seq = tokenizer.texts_to_sequences([caption])[0]
    return pad_sequences([seq], maxlen=max_len, padding="post")[0]
