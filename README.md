# Arabic Morphological Word Embeddings

A research codebase for training and evaluating **morphology-aware word embeddings for Arabic**. Instead of treating each word as an atomic token, the model decomposes every word into its morphological components (root, pattern, affixes, part-of-speech) using [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) analyses, and fuses them with a character-level CNN. This makes the embeddings robust to Arabic's rich, templatic morphology and to out-of-vocabulary (OOV) words.

The embeddings are trained with **Skip-Gram with Negative Sampling (SGNS)** plus a Wang & Isola uniformity regularizer, and evaluated on analogy and intrinsic similarity tasks.

---

## Model Architecture

Each word is encoded through several parallel pathways and fused into a single dense vector:

```
                ┌─ root embedding ──┐
                ├─ pattern embedding┤
word record ───▶├─ affix embedding ─┤──▶ Transformer fuser ──┐
                └─ POS embedding ───┘        +               ├─▶ morph vector
                                       linear projection ────┘        │
                                                                      │  fusion
  surface form ──▶ character CNN (multi-kernel) ─────────▶ char vector┘     │
                                                                            ▼
                                                            L2-normalized embedding
```

Two fusion strategies are provided as drop-in model variants:

| Variant | Folder | Fusion rule |
|---------|--------|-------------|
| **Residual Fusion** | `Core_Model/Model_w_Residual_Fusion/` | `v_final = v_morph + char_contribution` — the char-CNN adds a residual correction on top of the morphological vector. |
| **Interpolative Fusion** | `Core_Model/Model_w_Interpolative_Fusion/` | `v_final = (1−β)·v_morph + β·v_char` — a learned convex blend; OOV words let the CNN take full control (weight → 1.0). |

Both share the same `MorphoEmbeddingConfig` (component dim 128, projection dim 256, 4 attention heads, 2 encoder layers, char kernels `[2,3,4,5]`).

---

## Repository Layout

```
code/
├── Core_Model/                       # Model + training
│   ├── train_skipgram.py             # SGNS trainer (OneCycle LR, uniformity loss)
│   ├── morpho_dataset.py             # Streaming & word-level datasets
│   ├── cached_dataset.py             # Fast dataset from a pre-built .pt cache
│   ├── vocab_builder.py              # Multi-track vocabulary builder
│   ├── Model_w_Residual_Fusion/
│   │   └── morpho_embedding_model.py
│   └── Model_w_Interpolative_Fusion/
│       └── morpho_embedding_model.py
│
├── Data_Preprocessing/               # Corpus prep
│   ├── preprocess_combine.py         # Clean + merge Wikipedia + CC-100 → 2M sentences
│   ├── build_cache.py                # Tokenize/analyze corpus into a .pt training cache
│   └── extract_eval_words.py         # Pull clean eval words from analyzed data
│
├── Evaluation/
│   ├── intrinsic_eval.py             # Nearest-neighbor / similarity probes
│   └── evaluate_embeddings.py        # Analogy evaluation (with Arabic display helpers)
│
├── Vocabs/                           # Pre-built vocabularies (JSON)
│   ├── root_vocab.json     (3,860)
│   ├── pattern_vocab.json  (2,296)
│   ├── affix_vocab.json      (341)
│   ├── pos_vocab.json         (27)
│   ├── char_vocab.json     (2,627)
│   └── vocab_meta.json
│
└── Results_MGAD/                     # Analogy results on the MGAD benchmark
    └── analogy_results_Run_{G,H,I,J}.json
```

---

## Data

The training corpus is **~2M cleaned Arabic sentences**, built by `preprocess_combine.py` from:

- **~1M sentences** from Arabic Wikipedia
- **~500k sentences** from a CC-100 / web crawl slice

Cleaning rejects English-heavy lines, wiki/HTML/BBCode markup, URLs, timestamps, and any line that is not at least 70% Arabic characters. The cleaned text is then morphologically analyzed (CAMeL Tools) and packed into a `.pt` cache via `build_cache.py` for fast training.

> **Note:** The raw corpora, the `.pt` caches, and trained checkpoints are **not** included in this repo (see `.gitignore`). Only source code, vocabularies, and evaluation results are tracked.

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch tqdm numpy camel-tools arabic-reshaper python-bidi

# 2. Build the training corpus (expects data/wiki_1m.txt and data/cc100_500k.txt)
python Data_Preprocessing/preprocess_combine.py

# 3. Build the fast .pt cache from analyzed records
python Data_Preprocessing/build_cache.py --input data/camel_2m.jsonl --output data/cache.pt

# 4. Train (point the trainer at one of the two model variants)
python Core_Model/train_skipgram.py

# 5. Evaluate
python Evaluation/evaluate_embeddings.py     # analogy
python Evaluation/intrinsic_eval.py          # intrinsic similarity
```

Training hyperparameters live in the `TrainerConfig` dataclass in `train_skipgram.py` (epochs, batch size, LR schedule, window size, negative samples, uniformity weight, etc.).

---

## Results

Analogy accuracy on the **MGAD** benchmark is reported in `Results_MGAD/`. Each run records `accuracy_at_1`, `accuracy_at_5`, and a per-category breakdown (nominal vs. verbal analogies), along with how many questions were skipped as OOV. A representative run:

| Metric | Value |
|--------|-------|
| Answered / total | 3,943 / 20,001 |
| Accuracy@1 | 18.8% |
| Accuracy@5 | 48.3% |
| Nominal Acc@1 / @5 | 46.4% / 66.9% |
| Verb Acc@1 / @5 | 5.9% / 39.6% |

---

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.

Copyright © 2026 Obada Hamdan.
