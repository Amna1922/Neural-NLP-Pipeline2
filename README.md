# NLP Assignment 2: From-Scratch Implementation in PyTorch

This repository contains a complete implementation of a neural NLP pipeline from scratch in PyTorch, including word embeddings (TF-IDF, PPMI, Skip-gram Word2Vec), sequence labeling (POS tagging & NER with BiLSTM-CRF), and Transformer-based topic classification.

## Assignment Overview

This assignment implements three main components without using any pretrained models, Gensim, HuggingFace, or PyTorch's built-in Transformer classes:

| Part | Task | Key Components |
|------|------|----------------|
| Part 1 | Word Embeddings | TF-IDF, PPMI, Skip-gram Word2Vec |
| Part 2 | Sequence Labeling | BiLSTM with CRF for NER, BiLSTM for POS |
| Part 3 | Topic Classification | Transformer Encoder (from scratch) |

## Repository Structure

```
├── part1_word_embeddings.py    # TF-IDF, PPMI, Skip-gram implementation
├── part2_sequence_labeling.py  # POS tagging & NER with BiLSTM-CRF
├── part3_transformer.py        # Transformer encoder for classification
├── ner_standalone_fix.py       # Standalone NER fix with class weights
├── cleaned.txt                 # Preprocessed corpus (required)
├── raw.txt                     # Raw corpus for ablation (required)
├── Metadata.json               # Article metadata for categorization
└── README.md                   # This file
```

## Requirements

```bash
pip install torch numpy matplotlib scikit-learn
```

**System Requirements:**
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- CPU or CUDA-capable GPU

## Input File Format

### cleaned.txt & raw.txt

Each file contains documents separated by `[N]` markers:

```
[1]
Document text content here...
[2]
Another document content...
```

### Metadata.json

```json
{
  "1": {
    "title": "Article Title",
    "publish_date": "2026-01-31",
    "url": "https://..."
  }
}
```

## Reproduction Instructions

### Step 1: Prepare Data

Place the required files in the same directory as the scripts:
- `cleaned.txt` - Main training corpus (200+ documents)
- `raw.txt` - Raw corpus for ablation baseline
- `Metadata.json` - Topic labels for classification

### Step 2: Run Part 1 - Word Embeddings

```bash
python part1_word_embeddings.py
```

**Expected outputs:**
- `tfidf_matrix.npy` - TF-IDF weighted term-document matrix
- `ppmi_matrix.npy` - PPMI weighted co-occurrence matrix
- `embeddings_w2v.npy` - Skip-gram averaged embeddings (100d)
- `embeddings_c2_raw.npy` - Skip-gram on raw.txt (100d)
- `embeddings_c4_d200.npy` - Skip-gram with d=200
- `tsne_ppmi.png` - t-SNE visualization of top 200 tokens
- `w2v_loss_c3.png`, `w2v_loss_c2.png`, `w2v_loss_c4.png` - Training loss curves

**Console output includes:**
- Top-10 TF-IDF words per category
- PPMI nearest neighbors for query words
- Skip-gram training progress (loss per epoch)
- Top-10 nearest neighbors for Urdu query words
- Analogy test results
- Four-condition comparison (C1-C4) with MRR scores

> **Note:** Skip-gram training on CPU may take 2-4 hours depending on corpus size.

### Step 3: Run Part 2 - Sequence Labeling

```bash
python part2_sequence_labeling.py
```

**Expected outputs:**
- `loss_POS_frozen.png` - Training curve for frozen embeddings
- `loss_POS_finetuned.png` - Training curve for fine-tuned embeddings
- `loss_NER_CRF.png` - Training curve for NER with CRF
- `loss_NER_noCRF.png` - Training curve for NER without CRF

**Console output includes:**
- Dataset sizes and tag distributions
- POS tagging accuracy and macro-F1 for frozen/fine-tuned
- Confusion matrices for both POS configurations
- NER entity-level F1 scores
- Ablation study results (A1-A4)

> **Note:** This script runs 30+ epochs and may take 1-2 hours on CPU.

### Step 4: Fix NER (If Needed)

If NER shows 0% F1, run the standalone fix:

```bash
python ner_standalone_fix.py
```

This script adds class weights and data augmentation to improve NER performance.

### Step 5: Run Part 3 - Transformer Classification

```bash
python part3_transformer.py
```

**Expected outputs:**
- `transformer_training.png` - Training/validation loss and accuracy curves
- `attn_sample*_head*.png` - Attention heatmaps (6 images)

**Console output includes:**
- Vocabulary size and class distribution
- Training progress with learning rate schedule
- Test accuracy and macro-F1
- 5×5 confusion matrix
- BiLSTM vs Transformer comparison analysis

> **Note:** Transformer training on CPU takes 30-60 minutes for 20 epochs.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `TypeError: TSNE.__init__() got unexpected keyword argument 'n_iter'` | Change `n_iter=1000` to `max_iter=1000` in part1 |
| `UnicodeDecodeError` when loading JSON | Add `encoding='utf-8'` to `open()` calls |
| NER shows 0% F1 | Run `ner_standalone_fix.py` for class-weighted training |
| Matplotlib Arabic font warnings | Ignore - images still save correctly |
| Out of memory error | Reduce batch size in DataLoader (32→16 or 8) |
| Slow training | Use GPU: add `device = 'cuda'` or reduce epochs |

## Performance Tips

**For faster Part 1 training:**
- Reduce vocabulary size from 10000 to 5000
- Decrease Skip-gram epochs from 5 to 3
- Use smaller batch size (256 instead of 512)

**For faster Part 2 training:**
- Reduce dataset size from 500 to 300 sentences
- Decrease hidden_dim from 128 to 64
- Use fewer LSTM layers (1 instead of 2)

**For faster Part 3 training:**
- Reduce max_len from 256 to 128
- Decrease d_model from 128 to 64
- Use 2 encoder blocks instead of 4

## Expected Results Summary

| Component | Metric | Expected Score |
|-----------|--------|----------------|
| POS Tagging (frozen) | Accuracy | 75-80% |
| POS Tagging (fine-tuned) | Accuracy | 90-95% |
| NER (LOC) | F1 Score | 80-85% |
| Topic Classification | Accuracy | 55-65% |

## Citation

If you use this code for academic purposes, please cite accordingly:

```bibtex
@misc{nlp-assignment2,
  author = {Amna Zubair},
  title = {From-Scratch NLP Pipeline in PyTorch},
  year = {2026},
  url = {[repository-url]}
}
```

## License

This code is for educational purposes as part of CS-4063 Natural Language Processing course at FAST NUCES.

## Acknowledgments

- FAST NUCES CS-4063 Course Staff
- PyTorch documentation for reference
```
