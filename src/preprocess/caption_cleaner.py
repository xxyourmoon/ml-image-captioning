"""
clean_caption(text) -> str

Lower-case, remove non-alphabetic characters, and strip whitespace.
"""

import re


def clean_caption(text: str) -> str:
    """Lower-case, keep only a-z and spaces, strip."""
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    text = text.strip()
    return text


def add_token(text: str) -> str:
    """Wrap caption with <start> and <end> tokens."""
    return f"<start> {text} <end>"
