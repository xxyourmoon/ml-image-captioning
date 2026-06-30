# Image Captioning — CNN + LSTM / GRU

Final project for Machine Learning course.

**Tujuan:** Generate caption otomatis dari gambar menggunakan arsitektur CNN encoder + RNN decoder (LSTM dengan Attention, dan GRU sebagai baseline).

**Dataset:** [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — 8.091 gambar, 5 caption per gambar.

---

## Struktur Project

```
ml-image-captioning/
├── dataset/
│   ├── Images/              # 8.091 gambar JPEG
│   └── captions.txt         # 40.455 baris caption
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Eksplorasi dataset
│   ├── 02_data_preprocessing.ipynb   # Cleaning caption + token
│   ├── 03_feature_extraction.ipynb   # Ekstraksi fitur ResNet50
│   ├── 04_tokenizer.ipynb            # Tokenizer & vocabulary
│   ├── 05_data_loader.ipynb          # Data loader & train/val split
│   ├── 06_training_lstm.ipynb        # Training CNN+LSTM+Attention
│   ├── 07_training_gru.ipynb         # Training CNN+GRU (baseline)
│   ├── 08_evaluation.ipynb           # Evaluasi BLEU
│   └── 09_inference.ipynb            # Inference & demo
├── src/
│   ├── config.py                     # Path & hyperparameter global
│   ├── preprocess/
│   │   ├── caption_cleaner.py        # Lowercase, regex, strip
│   │   └── tokenizer.py              # Build/save/load tokenizer
│   ├── feature/
│   │   └── extractor.py              # ResNet50 feature extraction
│   ├── models/
│   │   ├── attention.py              # Bahdanau attention layer
│   │   ├── lstm_model.py             # CNN + LSTM + Attention
│   │   └── gru_model.py              # CNN + GRU (baseline)
│   └── evaluation/
│       └── metrics.py                # BLEU-1 s.d. BLEU-4
├── app/
│   └── app.py                        # Streamlit demo
├── model/
│   ├── features.pkl                  # Feature vectors (8.091 × 2048)
│   └── tokenizer.pkl                 # Tokenizer vocabulary
├── outputs/
│   ├── training/                     # History, weights (.keras)
│   ├── evaluation/                   # BLEU scores (.csv)
│   ├── inference/                    # Sample predictions
│   └── figures/                      # Plot training curves
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment
├── PROJECT_RULES.md
├── MASTER_AGENT_PROMPT.md
├── NOTEBOOK_TEMPLATE.md
└── GPU_SETUP.md
```

## Pipeline

```
Dataset → Preprocessing → Feature Extraction → Tokenizer
    → Data Loader → Training (LSTM / GRU) → Evaluation → Inference → Deploy
```

## Setup

```bash
# 1. Clone
git clone <repo>
cd ml-image-captioning

# 2. Conda environment
conda env create -f environment.yml
conda activate image-caption

# 3. Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 4. Run notebooks
jupyter notebook notebooks/
```

## GPU

Project wajib dijalankan di GPU (RTX 4060). Lihat `GPU_SETUP.md` untuk detail.

## Arsitektur

| Model | Encoder | Decoder | Attention |
|-------|---------|---------|-----------|
| **Primary** | ResNet50 | LSTM | Bahdanau |
| **Baseline** | ResNet50 | GRU | — |
