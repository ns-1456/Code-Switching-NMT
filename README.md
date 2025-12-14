# English → Hinglish Neural Machine Translation

A **from-scratch Transformer** encoder-decoder for translating English into Hinglish (Romanized Hindi-English code-mixed text), trained on 150k+ parallel sentence pairs.

## Results

| Metric | Score |
|--------|-------|
| **BLEU** | **58.35** |
| **chrF** | **74.69** |
| Test Set | 9,244 sentences |
| Training | 20 epochs on Google Colab Pro (T4) |

### Sample Translations

| English | Hinglish Output |
|---------|----------------|
| I am going home | mai ghar jaa raha hoon |
| Are you crazy? | Kya tum crazy ho? |
| What is your name? | aap ka naam kya hai? |
| Can you help me with this? | Kya aap muje is ke sath help kar sakte he? |
| I need to buy some groceries | mujhe kuch groceries kharidne ki zarurat hai |
| My phone is not working | mera phone nahi chal raha hai |
| He is my best friend | Wo mera best friend hai |
| I forgot my wallet at home | mai ghar me apna wallet bhul gaya hun |

### Attention Visualizations

Cross-attention heatmaps showing which English source words the model attends to when generating each Hinglish token (averaged across all 8 attention heads in the last decoder layer):

<p align="center">
  <img src="assets/attn_going_home.png" width="48%" />
  <img src="assets/attn_are_you_crazy.png" width="48%" />
</p>
<p align="center">
  <img src="assets/attn_lets_meet.png" width="48%" />
</p>

The model learns meaningful word alignments: "home" maps strongly to "ghar", "crazy" maps to "crazy" (code-switch retention), and "meet" aligns with "milne". The attention patterns confirm the model isn't just memorizing sequences — it's learning cross-lingual structure.

---

## Highlights

- **From-scratch Transformer** built entirely in PyTorch (no pre-trained weights, no HuggingFace model classes)
- **Custom BPE tokenizer** trained on mixed English + Hinglish corpus (16k shared vocab)
- **58.35 BLEU** on 9.2k test sentences (corpus-level, sacrebleu)
- **Attention visualization** showing learned word-level alignment between source and target
- **Beam search** decoding with configurable width
- **Streamlit demo** for interactive translation

## Architecture

| Component | Specification |
|-----------|---------------|
| Type | Encoder-Decoder Transformer (Seq2Seq) |
| Encoder | 4 layers, pre-norm, 8 heads |
| Decoder | 4 layers, pre-norm, 8 heads |
| d_model | 256 |
| d_ff | 1024 |
| Dropout | 0.15 |
| Vocab | 16,000 (shared BPE) |
| Parameters | ~15-20M |

All components hand-written in PyTorch: sinusoidal positional encoding, multi-head attention, feed-forward layers, encoder/decoder stacks with pre-norm residual connections, and mask utilities.

## Dataset

[`findnitai/english-to-hinglish`](https://huggingface.co/datasets/findnitai/english-to-hinglish) — 189k parallel English-Hinglish pairs from HuggingFace (Apache 2.0).

**Cleaning pipeline** (reduces to ~170k):
- Remove rows with Devanagari script (model is Romanized-only)
- Length filter: 3-50 words per sentence
- Normalize: lowercase, strip URLs/handles/hashtags/emojis
- Deduplicate exact pairs
- Split: 90/5/5 train/val/test

## Project Structure

```
├── src/
│   ├── data_pipeline.py     # Download, clean, split dataset
│   ├── tokenizer.py         # Train BPE tokenizer from scratch
│   ├── model.py             # Full Transformer (from scratch in PyTorch)
│   ├── train.py             # Training loop with teacher forcing
│   ├── inference.py         # Greedy + beam search decoding
│   └── evaluate.py          # BLEU, chrF, qualitative evaluation
├── app.py                   # Streamlit demo
├── notebooks/
│   └── colab_train.ipynb    # Master training notebook for Colab Pro
├── configs/
│   └── config.yaml          # All hyperparameters
├── assets/                  # Attention heatmap images
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
# Step 1: Download and clean data
python -m src.data_pipeline

# Step 2: Train tokenizer
python -m src.tokenizer

# Step 3: Train model
python -m src.train

# Step 4: Evaluate
python -m src.evaluate
```

### 3. Launch demo

```bash
streamlit run app.py
```

### Google Colab

Open `notebooks/colab_train.ipynb` in Colab Pro and run all cells. Training takes ~2-3 hours on a T4 GPU.

## Training Details

- **Optimizer**: AdamW (lr=3e-4, betas=(0.9, 0.98), weight_decay=0.01)
- **Scheduler**: Linear warmup (1000 steps) + cosine decay to 1e-5
- **Loss**: Cross-entropy with label smoothing (0.1)
- **Regularization**: Dropout (0.15), gradient clipping (max_norm=1.0)
- **Teacher forcing**: Ground-truth previous token fed to decoder during training
- **Early stopping**: Patience 3 on validation loss
- **Batch size**: 128
- **Epochs**: 20 (ran full, no early stop triggered)

## Evaluation

| Metric | Score | Notes |
|--------|-------|-------|
| BLEU | 58.35 | Corpus-level via sacrebleu |
| chrF | 74.69 | Character-level F-score, better for morphologically rich Hinglish |

**Known limitation**: Some common English phrases (e.g., "Don't worry about it") are passed through verbatim. This is a known challenge in code-mixed NMT — since Hinglish naturally contains full English phrases, the model sometimes learns that copying is a valid translation strategy.

## License

Apache 2.0 (dataset license).
