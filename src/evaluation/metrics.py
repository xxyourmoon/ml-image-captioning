"""
Evaluation metrics for caption generation.

Uses NLTK's BLEU score implementation.
"""

from typing import List, Dict

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction


def compute_bleu(references: List[List[str]],
                 hypotheses: List[str]) -> Dict[str, float]:
    """
    Corpus-level BLEU-1 through BLEU-4.

    Args:
        references: list of reference captions per image (list of strings)
        hypotheses: list of generated captions (strings)

    Returns:
        dict with keys "bleu_1", "bleu_2", "bleu_3", "bleu_4"
    """
    smoothie = SmoothingFunction().method4

    refs_tokenized = [[ref.split() for ref in ref_group] for ref_group in references]
    hyps_tokenized = [hyp.split() for hyp in hypotheses]

    bleu_scores = {}
    for n in range(1, 5):
        score = corpus_bleu(
            refs_tokenized,
            hyps_tokenized,
            weights=tuple(1.0 / n for _ in range(n)),
            smoothing_function=smoothie,
        )
        bleu_scores[f"bleu_{n}"] = round(score, 4)

    return bleu_scores


def format_bleu_report(bleu: Dict[str, float]) -> str:
    """Pretty-print BLEU scores."""
    return " | ".join(f"{k.upper()}: {v:.4f}" for k, v in bleu.items())
