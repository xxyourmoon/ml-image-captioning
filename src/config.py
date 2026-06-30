"""
Shared configuration for the project.

Centralises paths, random seeds, and model hyper-parameters
so every notebook and module reads from a single source of truth.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR   = PROJECT_ROOT / "dataset"
IMAGES_DIR    = DATASET_DIR / "Images"
CAPTIONS_PATH = DATASET_DIR / "captions.txt"

MODEL_DIR     = PROJECT_ROOT / "model"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
TRAINING_DIR  = OUTPUTS_DIR / "training"
EVAL_DIR      = OUTPUTS_DIR / "evaluation"
INFERENCE_DIR = OUTPUTS_DIR / "inference"
FIGURES_DIR   = OUTPUTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

IMG_SIZE = (224, 224)   # ResNet50 input size

# ---------------------------------------------------------------------------
# Caption / Tokenizer
# ---------------------------------------------------------------------------

VOCAB_SIZE     = 5000
MAX_CAPTION_LEN = 34

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

BATCH_SIZE      = 64
EPOCHS          = 50
EMBEDDING_DIM   = 256
LSTM_UNITS      = 512
GRU_UNITS       = 512
LEARNING_RATE   = 1e-3
DROPOUT_RATE    = 0.5

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

USE_MIXED_PRECISION = True
ENABLE_MEMORY_GROWTH = True
