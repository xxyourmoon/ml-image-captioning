# PROJECT RULES

## Project Name

Image Captioning using CNN-LSTM and CNN-GRU

---

# PROJECT OBJECTIVE

Build an end-to-end machine learning pipeline that generates natural language captions from images.

The project is intended as:
- Final project for Machine Learning course
- First step toward Vision Language Model understanding
- Practical and reproducible implementation

---

# SUCCESS CRITERIA

The project is complete if:

[ ] Dataset downloaded and validated
[ ] Preprocessing completed
[ ] Feature extraction completed
[ ] Tokenizer completed
[ ] CNN+LSTM trained
[ ] CNN+GRU trained
[ ] BLEU evaluation completed
[ ] Inference notebook works
[ ] Streamlit app runs
[ ] Documentation completed

---

# DATASET

Dataset:
Flickr8k

Source:
https://www.kaggle.com/datasets/adityajn105/flickr8k

Structure:

dataset/

Images/
captions.txt

---

# MODEL SCOPE

Primary:
CNN + LSTM + Attention

Baseline:
CNN + GRU

Do NOT use:
- Transformer
- ViT
- BLIP
- CLIP
- LLaVA
- Large VLM

---

# COMPUTE POLICY

Training MUST use NVIDIA GPU.

Do NOT intentionally use CPU.

Use:
TensorFlow GPU

Validate GPU before training.

Required:

tf.config.list_physical_devices()

Expected:

GPU available

Mixed precision allowed.

Enable memory growth.

---

# CODE STYLE

Requirements:

- Modular
- Typed
- Reproducible
- Notebook + src compatible

Naming:

snake_case

Notebook:

NN_title.ipynb

Example:

01_dataset_exploration.ipynb

---

# NOTEBOOK POLICY

Every notebook MUST contain:

1 Purpose

2 Theory

3 Implementation

4 Output

5 Observation

6 Next Step

---

# OUTPUT DIRECTORY

outputs/

training/

evaluation/

inference/

figures/

---

# REPRODUCIBILITY

Random seed:

42

Framework:

TensorFlow

Environment:

Conda

---

# END GOAL

Upload image
↓

Generate caption

↓

Evaluate BLEU

↓

Deploy Streamlit