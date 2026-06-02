from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



# Arabic terminal display helper

def ar(text: str) -> str:
    """Reshape and apply bidi to Arabic text for correct terminal display."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(str(text)))
    except ImportError:
        return str(text)


# Lazy imports for optional dependencies
def _require(package: str, install: str):
    try:
        return __import__(package)
    except ImportError:
        raise ImportError(
            f"'{package}' is required for this feature. "
            f"Install with:  pip install {install}"
        )


# 1. EmbeddingStore
class EmbeddingStore:

    def __init__(
        self,
        checkpoint_path: str,
        vocab_dir:       str,
        word_list_path:  Optional[str] = None,
        max_word_len:    int = 20,
        device:          str = "cpu",
    ) -> None:
        self.device      = torch.device(device)
        self.max_word_len = max_word_len

        # Load vocab
        from vocab_builder import VocabBuilder
        self.vocab = VocabBuilder.load(Path(vocab_dir))
        logger.info("Vocab loaded from %s", vocab_dir)

        # Load model 
        from morpho_embedding_model import MorphoEmbeddingModel, MorphoEmbeddingConfig
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg   = state["model_cfg"]
        self.model = MorphoEmbeddingModel(cfg=cfg).to(self.device)
        self.model.load_state_dict(state["encoder_state"])
        self.model.eval()
        logger.info(
            "Model loaded from %s  (epoch %d, loss %.4f)",
            checkpoint_path, state.get("epoch", "?"), state.get("loss", float("nan")),
        )

        #  Build index 
        self.words:   list[str]        = []
        self.records: list[dict]       = []
        self.vectors: Optional[torch.Tensor] = None  # (N, proj_dim)

        if word_list_path:
            self._build_index(word_list_path)

    # Index building
    def _build_index(self, path: str, batch_size: int = 256) -> None:
        """Embed every word in the word list and store vectors in RAM."""
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        logger.info("Embedding %d words...", len(records))
        all_vecs = []

        with torch.no_grad():
            for i in range(0, len(records), batch_size):
                batch   = records[i : i + batch_size]
                morpho, chars, oov = self._encode_batch(batch)
                vecs    = self.model(morpho, chars, oov)   # (B, D) L2-normed
                all_vecs.append(vecs.cpu())

        self.words   = [r["word"] for r in records]
        self.records = records
        self.vectors = torch.cat(all_vecs, dim=0)   # (N, D)
        logger.info("Index built: %d vectors of dim %d", len(self.words), self.vectors.shape[1])

    def _encode_batch(
        self, batch: list[dict]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Encode a list of word records into model-ready tensors."""
        morpho_lists = {"root": [], "pattern": [], "affix": [], "pos": []}
        char_list    = []
        oov_list     = []

        for record in batch:
            enc  = self.vocab.encode_record(record)
            chars = self.vocab.encode_chars(record.get("word", ""), self.max_word_len)

            for k in morpho_lists:
                morpho_lists[k].append(enc[k])
            char_list.append(chars)
            oov_list.append(bool(enc["is_oov"]))

        morpho = {
            k: torch.tensor(v, dtype=torch.long, device=self.device)
            for k, v in morpho_lists.items()
        }
        chars = torch.tensor(char_list, dtype=torch.long, device=self.device)
        oov   = torch.tensor(oov_list,  dtype=torch.bool,  device=self.device)
        return morpho, chars, oov

    # Single-word embedding
    def embed_word(self, record: dict) -> torch.Tensor:

        morpho, chars, oov = self._encode_batch([record])
        with torch.no_grad():
            vec = self.model(morpho, chars, oov)   # (1, D)
        return vec.squeeze(0).cpu()

    # Similarity search
    def most_similar(
        self,
        query_vec: torch.Tensor,
        top_k:     int = 10,
        exclude:   Optional[list[str]] = None,
    ) -> list[tuple[str, float]]:

        if self.vectors is None:
            raise RuntimeError("No index built. Pass word_list_path to __init__.")

        exclude_set = set(exclude or [])
        # Cosine similarity: both query and index are L2-normed, so dot = cos
        sims = (self.vectors @ query_vec).tolist()   # (N,)

        ranked = sorted(
            [(self.words[i], sims[i]) for i in range(len(self.words))
             if self.words[i] not in exclude_set],
            key=lambda x: -x[1],
        )
        return ranked[:top_k]

    def similar_to_word(
        self, record: dict, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Convenience: embed a word then find its nearest neighbours."""
        vec = self.embed_word(record)
        return self.most_similar(vec, top_k=top_k, exclude=[record["word"]])

    def similar_to_root(
        self, root: str, top_k: int = 10
    ) -> list[tuple[str, float]]:

        if self.vectors is None:
            raise RuntimeError("No index built.")

        def _strip(r: str) -> str:
            return r.replace(".", "").replace("-", "").strip()

        root_stripped = _strip(root)

        indices = [
            i for i, r in enumerate(self.records)
            if _strip(r.get("root", "")) == root_stripped
        ]
        if not indices:
            logger.warning(
                "Root '%s' (stripped: '%s') not found in index.",
                root, root_stripped,
            )
            return []

        logger.info(
            "Root '%s' — found %d matching words in index.",
            root, len(indices),
        )
        root_vecs = self.vectors[indices]
        centroid  = F.normalize(
            root_vecs.mean(dim=0, keepdim=True), dim=-1
        ).squeeze(0)
        return self.most_similar(centroid, top_k=top_k)


# 2. Analogy evaluation
def analogy(
    store:   EmbeddingStore,
    word_a:  dict,
    word_b:  dict,
    word_c:  dict,
    top_k:   int = 5,
) -> list[tuple[str, float]]:

    v_a = store.embed_word(word_a)
    v_b = store.embed_word(word_b)
    v_c = store.embed_word(word_c)

    query = F.normalize((v_a - v_b + v_c).unsqueeze(0), dim=-1).squeeze(0)

    exclude = [word_a["word"], word_b["word"], word_c["word"]]
    return store.most_similar(query, top_k=top_k, exclude=exclude)


# 3. Gate inspection
def inspect_gates(checkpoint_path: str) -> dict[str, float]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd    = state["encoder_state"]

    alpha_raw = sd["alpha_gate"].item()
    beta_raw  = sd["beta_base"].item()

    alpha = torch.sigmoid(torch.tensor(alpha_raw)).item()
    beta  = torch.sigmoid(torch.tensor(beta_raw)).item()

    results = {
        "alpha_gate_raw":        round(alpha_raw, 6),
        "alpha_gate_activated":  round(alpha,     6),
        "beta_base_raw":         round(beta_raw,  6),
        "beta_base_activated":   round(beta,      6),
        "interpretation_alpha":  (
            "Transformer-dominant" if alpha > 0.6
            else "Linear-dominant" if alpha < 0.4
            else "Balanced"
        ),
        "interpretation_beta":   (
            "CNN light stabiliser (ideal)"     if beta < 0.2
            else "CNN moderate contributor"    if beta < 0.5
            else "CNN dominant (check morpho quality)"
        ),
    }

    logger.info("─── Gate Inspection ──────────────────────────────────")
    logger.info("  alpha_gate  raw=%.4f  sigmoid=%.4f  → %s",
                alpha_raw, alpha, results["interpretation_alpha"])
    logger.info("  beta_base   raw=%.4f  sigmoid=%.4f  → %s",
                beta_raw, beta, results["interpretation_beta"])
    logger.info("──────────────────────────────────────────────────────")

    return results


# 4. OOV stress test
def oov_stress_test(
    store:          EmbeddingStore,
    clean_record:   dict,
    typo_records:   list[dict],
) -> list[dict]:
    v_clean = store.embed_word(clean_record)   # (D,)

    results = []
    logger.info("OOV Stress Test: '%s'", ar(clean_record["word"]))
    logger.info("  %-25s  %-8s  %-8s  %-6s", "word", "cos_sim", "L2_dist", "oov")

    for rec in typo_records:
        enc    = store.vocab.encode_record(rec)
        is_oov = bool(enc["is_oov"])
        v_typo = store.embed_word(rec)

        cos_sim = F.cosine_similarity(v_clean.unsqueeze(0),
                                      v_typo.unsqueeze(0)).item()
        l2_dist = (v_clean - v_typo).norm().item()

        logger.info(
            "  %-25s  %-8.4f  %-8.4f  %-6s",
            ar(rec["word"]), cos_sim, l2_dist, "OOV" if is_oov else "IV",
        )
        results.append({
            "word":    rec["word"],
            "is_oov":  is_oov,
            "cos_sim": round(cos_sim, 4),
            "l2_dist": round(l2_dist, 4),
        })

    logger.info("──────────────────────────────────────────────────────")
    return results


# 5. Visualisation — t-SNE / PCA
def visualise(
    store:        EmbeddingStore,
    output_path:  str,
    method:       str = "tsne",
    color_by:     str = "root",
    highlight_roots: Optional[list[str]] = None,
    max_words:    int = 500,
    perplexity:   int = 30,
    random_state: int = 42,
) -> None:
    plt = _require("matplotlib.pyplot", "matplotlib")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if method == "tsne":
        from sklearn.manifold import TSNE
    else:
        from sklearn.decomposition import PCA

    if store.vectors is None:
        raise RuntimeError("No index built.")

    # Subsample if needed
    n = min(max_words, len(store.words))
    idx = np.random.RandomState(random_state).choice(len(store.words), n, replace=False)
    vecs    = store.vectors[idx].numpy()               # (n, D)
    words   = [store.words[i]   for i in idx]
    records = [store.records[i] for i in idx]

    # Reduce to 2D
    logger.info("Running %s on %d vectors...", method.upper(), n)
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=perplexity,
                       random_state=random_state, n_iter=1000,
                       init="pca", learning_rate="auto")
    else:
        reducer = PCA(n_components=2, random_state=random_state)

    coords = reducer.fit_transform(vecs)   # (n, 2)

    # Color assignment
    color_keys = [r.get(color_by, "?") for r in records]
    unique_keys = sorted(set(color_keys))
    cmap        = plt.get_cmap("tab20", max(len(unique_keys), 1))
    key_to_color = {k: cmap(i) for i, k in enumerate(unique_keys)}
    colors       = [key_to_color[k] for k in color_keys]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=18, alpha=0.7,
               linewidths=0.3, edgecolors="white")

    # Label highlighted roots
    highlight_set = set(highlight_roots or [])
    for i, rec in enumerate(records):
        if rec.get("root", "") in highlight_set or rec.get("word", "") in highlight_set:
            ax.annotate(
                words[i],
                (coords[i, 0], coords[i, 1]),
                fontsize=7,
                fontfamily="DejaVu Sans",
                ha="center",
                va="bottom",
                color="#111111",
                arrowprops=None,
            )

    # Legend (top 12 keys only to avoid clutter)
    legend_keys = unique_keys[:12]
    patches = [
        mpatches.Patch(color=key_to_color[k], label=k)
        for k in legend_keys
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7,
              title=color_by, title_fontsize=8, framealpha=0.8)

    title_method = "t-SNE" if method == "tsne" else "PCA"
    ax.set_title(
        f"Arabic Morphological Embeddings — {title_method} projection\n"
        f"MorphoEmbeddingModel  |  {n} words  |  colored by {color_by}",
        fontsize=11, pad=12,
    )
    ax.set_xlabel(f"{title_method}-1", fontsize=9)
    ax.set_ylabel(f"{title_method}-2", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved -> %s", output_path)


# Helper; extract word list from corpus
def _is_clean_token(tok: dict) -> bool:
    root    = tok.get("root", "") or ""
    pattern = (tok.get("lexeme_pattern") or tok.get("surface_pattern") or "")
    pos     = tok.get("pos", "") or ""

    if not root or not pattern or not pos:
        return False
    if len(root) not in (3, 4):
        return False
    if "#" in root or "#" in pattern:
        return False
    if root.upper() in ("NTWS", "FOREIGN", "NO_ANALYSIS"):
        return False
    if pattern.upper() in ("NTWS", "FOREIGN", "NO_ANALYSIS"):
        return False
    if pos in ("punc", "digit"):
        return False
    return True


def extract_eval_words(
    data_path:   str,
    output_path: str,
    n_sentences: int = 5_000,
) -> None:
    seen:         set[str]  = set()
    words:        list[dict] = []
    total_tokens: int = 0
    skipped:      int = 0

    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_sentences:
                break
            try:
                obj = json.loads(line)
                for tok in obj.get("tokens", []):
                    total_tokens += 1
                    word = tok.get("word", tok.get("surface", "")) or ""
                    if not word:
                        skipped += 1
                        continue
                    if not _is_clean_token(tok):
                        skipped += 1
                        continue
                    if word not in seen:
                        seen.add(word)
                        words.append({
                            "word":    word,
                            "root":    tok.get("root", "") or "",
                            "pattern": (tok.get("lexeme_pattern")
                                        or tok.get("surface_pattern") or ""),
                            "prefix":  "".join(tok.get("prefixes", [])),
                            "suffix":  "".join(tok.get("suffixes", [])),
                            "pos":     tok.get("pos", "") or "",
                        })
            except json.JSONDecodeError:
                continue

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in words:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    clean_pct = 100 * (total_tokens - skipped) / max(total_tokens, 1)
    logger.info(
        "Extracted %d unique clean words from %d tokens (%.1f%% clean) -> %s",
        len(words), total_tokens, clean_pct, output_path,
    )


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained Arabic Morphological Embeddings."
    )
    parser.add_argument("--checkpoint",  default="checkpoints/morpho_skipgram_best.pt")
    parser.add_argument("--vocab_dir",   default="vocabs/")
    parser.add_argument("--word_list",   default=None,
                        help="Path to eval_words.jsonl. "
                             "Generate with --extract_words first.")
    parser.add_argument("--output_dir",  default="eval_output/")
    parser.add_argument("--gate_only",   action="store_true",
                        help="Only print gate values, skip embedding eval.")
    parser.add_argument("--extract_words", action="store_true",
                        help="Extract word list from corpus and exit.")
    parser.add_argument("--data_path",   default="data/camel_2m.jsonl")
    parser.add_argument("--output",      default="data/eval_words.jsonl")
    parser.add_argument("--n_sentences", type=int, default=5_000)
    parser.add_argument("--tsne",        action="store_true",
                        help="Run t-SNE visualisation.")
    parser.add_argument("--pca",         action="store_true",
                        help="Run PCA visualisation.")
    parser.add_argument("--highlight",   nargs="*", default=["ك.ت.ب", "ج.م.ل", "ر.س.ل"],
                        help="Roots to highlight on the visualisation.")
    args = parser.parse_args()

    #  Step 0: extract word list from corpus 
    if args.extract_words:
        extract_eval_words(args.data_path, args.output, args.n_sentences)
        raise SystemExit(0)

    #  Step 1: gate inspection (no word list needed) 
    gates = inspect_gates(args.checkpoint)
    print("\n=== Gate Values ===")
    for k, v in gates.items():
        print(f"  {k:<30} {v}")

    if args.gate_only:
        raise SystemExit(0)

    #  Step 2: build embedding store 
    if not args.word_list:
        logger.warning(
            "No --word_list provided. "
            "Run with --extract_words first to generate one, "
            "then re-run with --word_list path/to/eval_words.jsonl"
        )
        raise SystemExit(1)

    store = EmbeddingStore(
        checkpoint_path=args.checkpoint,
        vocab_dir=args.vocab_dir,
        word_list_path=args.word_list,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    #  Step 3: analogy test 
    print("\n=== Analogy: v(kataba) - v(faala) + v(mafoul) ~ maktub ===")
    print(f"    {ar('كَتَبَ')} - {ar('فَعَلَ')} + {ar('مَفْعُول')} ~ {ar('مَكْتُوب')}")
    word_a = {"word": "كَتَبَ",  "root": "ك.ت.ب", "pattern": "فَعَلَ",
              "prefix": "", "suffix": "", "pos": "verb"}
    word_b = {"word": "فَعَلَ",  "root": "ف.ع.ل", "pattern": "فَعَلَ",
              "prefix": "", "suffix": "", "pos": "verb"}
    word_c = {"word": "مَفْعُول", "root": "ف.ع.ل", "pattern": "مَفْعُول",
              "prefix": "", "suffix": "", "pos": "adj"}

    results = analogy(store, word_a, word_b, word_c, top_k=5)
    for rank, (word, sim) in enumerate(results, 1):
        is_target = word in ("مَكْتُوب", "مكتوب")
        marker    = "  <-- TARGET" if is_target else ""
        print(f"  {rank}. {ar(word):<25} cos={sim:.4f}{marker}")

    #  Step 4: root similarity 
    for root in ["ك.ت.ب", "ج.م.ل", "ر.س.ل"]:
        print(f"\n=== Top 10 similar words for root {ar(root)} ===")
        neighbours = store.similar_to_root(root, top_k=10)
        if not neighbours:
            print(f"  (not found in index — try حكم or جمع which appear in your corpus)")
            continue
        for rank, (word, sim) in enumerate(neighbours, 1):
            print(f"  {rank:>2}. {ar(word):<25} cos={sim:.4f}")

    #  Step 5: OOV stress test 
    print(f"\n=== OOV Stress Test for {ar('مكتوب')} (maktub) ===")
    clean = {"word": "مكتوب", "root": "ك.ت.ب", "pattern": "مَفْعُول",
             "prefix": "", "suffix": "", "pos": "adj"}
    typos = [
        # In-vocabulary with correct morphology
        {"word": "مكتوب",  "root": "ك.ت.ب", "pattern": "مَفْعُول",
         "prefix": "", "suffix": "", "pos": "adj"},
        # OOV: morpho fields stripped (CNN must carry the full load)
        {"word": "مكتوب",  "root": "", "pattern": "",
         "prefix": "", "suffix": "", "pos": ""},
        # OOV: dialectal / alternate spelling
        {"word": "مكتوبة", "root": "", "pattern": "",
         "prefix": "", "suffix": "", "pos": ""},
        # OOV: deliberate letter swap (tests CNN robustness)
        {"word": "مكتوپ",  "root": "", "pattern": "",
         "prefix": "", "suffix": "", "pos": ""},
        # OOV: truncated word
        {"word": "مكتو",   "root": "", "pattern": "",
         "prefix": "", "suffix": "", "pos": ""},
    ]
    oov_results = oov_stress_test(store, clean, typos)

    # Save OOV results as JSON
    oov_path = Path(args.output_dir) / "oov_stress_test.json"
    with open(oov_path, "w", encoding="utf-8") as f:
        json.dump(oov_results, f, ensure_ascii=False, indent=2)
    logger.info("OOV results saved -> %s", oov_path)

    #  Step 6: visualisation 
    if args.tsne:
        visualise(
            store,
            output_path=str(Path(args.output_dir) / "tsne_by_root.png"),
            method="tsne",
            color_by="root",
            highlight_roots=args.highlight,
            max_words=500,
            perplexity=30,
        )
        visualise(
            store,
            output_path=str(Path(args.output_dir) / "tsne_by_pos.png"),
            method="tsne",
            color_by="pos",
            highlight_roots=args.highlight,
            max_words=500,
            perplexity=30,
        )

    if args.pca:
        visualise(
            store,
            output_path=str(Path(args.output_dir) / "pca_by_root.png"),
            method="pca",
            color_by="root",
            highlight_roots=args.highlight,
            max_words=500,
        )

    print(f"\nAll evaluation outputs saved to: {args.output_dir}")