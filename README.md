# CommonsenseQA – Natural Language Processing (Individual Project)

An individual Machine Learning project evaluating commonsense reasoning on the **CommonsenseQA** benchmark using custom **MLP** and **GRU (RNN)** architectures paired with static **FastText** embeddings.

---

## Project Context
- **Author:** Flora Gashi
- **Project Type:** Individual NLP Project / Assignment
- **Goal:** Classify the correct answer choice among 5 options (A–E) for commonsense reasoning questions using shallow neural architectures.

---

## Project Overview
- **Task:** 5-choice multiple-choice classification.
- **Baseline:** Random guessing yields **20.0% accuracy**[cite: 2].
- **Dataset:** Hugging Face `commonsense_qa`.

---

## Pipeline & Architecture Details

### 1. Preprocessing & FastText Embeddings
- **Embeddings:** Pretrained **FastText (300d)** vectors; fallback zero-vector for unknown/empty sequences[cite: 2].
- **Tokenization:** Lowercasing, whitespace tokenization, preserving apostrophes for contractions[cite: 2].
- **Logic Preservation:** **No stopword removal** (crucial to keep logical connectors intact)[cite: 2].

### 2. Model Architectures
- **MLP (Classifier):**
  - Input: Mean-pooled question ($300d$) concatenated with mean-pooled option ($300d$) $\rightarrow$ $600d$ vector per option[cite: 2].
  - Model: One hidden layer with ReLU and Dropout, processing each of the 5 choices independently to return 1 logit per answer[cite: 2].
- **RNN (GRU Classifier):**
  - Input: Concatenated Question + Answer token sequences (e.g., *"Where do we find magazines? bookstore"*) padded to batch max-length[cite: 2].
  - Model: **2-layer GRU** taking the final hidden state to output 1 logit per answer choice[cite: 2].

---

## Experiments & Hyperparameter Tuning

Grid search performed over **12 runs per model architecture** with the following setup[cite: 2]:
- **Learning Rates:** `1e-3`, `1e-4`, `1e-5`[cite: 2]
- **Dropout:** `0.1`, `0.3`[cite: 2]
- **Hidden Dimensions:** `128`, `256`[cite: 2]
- **Optimizer & Loss:** Adam Optimizer & CrossEntropyLoss (Batch Size: 128)[cite: 2]
- **Training:** Up to 40 epochs with Early Stopping (patience = 12 based on validation accuracy)[cite: 2].

---

## Results & Tracking

Experiment tracking, loss curves, and hyperparameter logs were recorded using **Weights & Biases**[cite: 2].

| Model | Random Baseline | Validation Accuracy | Test Accuracy |
|-------|-----------------|---------------------|---------------|
| **Random Guess** | 20.0% | - | 20.0% |
| **MLP (Classifier)** | 20.0% | **25.6%** | **26.9%** |
| **GRU (2-Layer RNN)** | 20.0% | 25.2% | 23.7% |

**Weights & Biases Dashboard:** [View W&B Runs & Experiments](https://wandb.ai/flora-gashi-hochschule-luzern/commonsense_project_1)[cite: 2]  
**Presentation:** [Download Presentation PDF](./presentation_commonsense_qa.pdf)

---

## Key Findings & Error Analysis

- **Performance:** The mean-pooled MLP slightly outperformed the sequential 2-layer GRU on the test set ($26.9\%$ vs. $23.7\%$)[cite: 2].
- **Distractor Confusion:** Error analysis on misclassified examples showed models often get confused between semantically similar options (e.g., *"make peace"* vs. *"make noise"*)[cite: 2].
- **Question Prefix Performance:**
  - Questions starting with **"WHERE"** achieved the highest accuracy[cite: 2].
  - Questions starting with **"WHEN"** were the most challenging and yielded the lowest accuracy[cite: 2].

---

## Reproduction & Setup

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone [TODO)
cd commonsense-qa-nlp
pip install -r requirements.txt
