from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DIACRITICS = set(
    "\u064B\u064C\u064D\u064E\u064F"
    "\u0650\u0651\u0652\u0653\u0654\u0655"
)


# Helpers
def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks for normalised lookup."""
    return "".join(c for c in text if c not in DIACRITICS)


def normalise(word: str) -> str:
    """Lowercase + strip diacritics for case-insensitive matching."""
    return strip_diacritics(word.strip())


# Embedding index
class EmbeddingIndex:

    def __init__(
        self,
        checkpoint_path: str,
        vocab_dir:        str,
        word_list_path:   Optional[str] = None,
        device:           str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        # Load vocab
        from vocab_builder import VocabBuilder
        self.vocab = VocabBuilder.load(Path(vocab_dir))
        logger.info("Vocab loaded from %s", vocab_dir)

        # Load model
        from morpho_embedding_model import MorphoEmbeddingModel
        state      = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)
        cfg        = state["model_cfg"]
        self.model = MorphoEmbeddingModel(cfg=cfg).to(self.device)
        self.model.load_state_dict(state["encoder_state"])
        self.model.eval()
        logger.info(
            "Model loaded  (epoch %s, loss %.4f)",
            state.get("epoch", "?"), state.get("loss", float("nan")),
        )

        # word → vector index
        self._norm_to_vec: dict[str, torch.Tensor] = {}
        self._norm_to_word: dict[str, str]          = {}

        if word_list_path:
            self._build_index(word_list_path)

    def _build_index(self, path: str, batch_size: int = 512) -> None:
        """Embed every word in the word list."""
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        logger.info("Building embedding index from %d records...", len(records))

        with torch.no_grad():
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                morpho, chars, oov = self._encode_batch(batch)
                vecs = self.model(morpho, chars, oov).cpu()   # (B, D)

                for j, rec in enumerate(batch):
                    word = rec.get("word", "") or ""
                    key  = normalise(word)
                    if key and key not in self._norm_to_vec:
                        self._norm_to_vec[key]  = vecs[j]
                        self._norm_to_word[key] = word

        logger.info("Index built: %d unique words", len(self._norm_to_vec))

    def _encode_batch(
        self, batch: list[dict]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        morpho_lists = {"root": [], "pattern": [], "affix": [], "pos": []}
        char_list, oov_list = [], []

        for rec in batch:
            enc   = self.vocab.encode_record(rec)
            chars = self.vocab.encode_chars(rec.get("word", ""), max_len=20)
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

    def get(self, word: str) -> Optional[torch.Tensor]:
        """Return the L2-normed vector for a word, or None if not in index."""
        return self._norm_to_vec.get(normalise(word))

    def embed_word_direct(self, record: dict) -> torch.Tensor:
        """Embed a single word record directly through the model."""
        morpho, chars, oov = self._encode_batch([record])
        with torch.no_grad():
            return self.model(morpho, chars, oov).squeeze(0).cpu()

    @property
    def all_vectors(self) -> torch.Tensor:
        """Stack all indexed vectors into a single matrix (N, D)."""
        return torch.stack(list(self._norm_to_vec.values()))

    @property
    def all_words(self) -> list[str]:
        return list(self._norm_to_word.values())

    def __len__(self) -> int:
        return len(self._norm_to_vec)


# 1. Morphological analogy evaluation
def evaluate_analogies(
    index:          EmbeddingIndex,
    analogy_path:   str,
    top_k:          int = 5,
    skip_oov:       bool = True,
) -> dict:
    # Load all vectors as a matrix for fast batch similarity
    all_vecs  = index.all_vectors    # (N, D)
    all_words = index.all_words

    # Build reverse map: normalised word → index in all_vecs
    word_to_idx = {normalise(w): i for i, w in enumerate(all_words)}

    questions_total  = 0
    questions_oov    = 0
    correct_at_1     = 0
    correct_at_k     = 0

    categories: dict[str, dict] = {}
    current_category = "general"

    with open(analogy_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Section header
            if line.startswith(":") or line.startswith("#"):
                current_category = line.lstrip(":# ").strip()
                if current_category not in categories:
                    categories[current_category] = {
                        "total": 0, "oov": 0,
                        "correct_at_1": 0, "correct_at_k": 0,
                    }
                continue

            parts = line.split()
            if len(parts) != 4:
                continue

            word_a, word_b, word_c, word_d = parts
            questions_total += 1

            if current_category not in categories:
                categories[current_category] = {
                    "total": 0, "oov": 0,
                    "correct_at_1": 0, "correct_at_k": 0,
                }
            categories[current_category]["total"] += 1

            # Check OOV
            keys = [normalise(w) for w in [word_a, word_b, word_c, word_d]]
            if skip_oov and any(k not in word_to_idx for k in keys):
                questions_oov += 1
                categories[current_category]["oov"] += 1
                continue

            # Get vectors
            try:
                v_a = all_vecs[word_to_idx[keys[0]]]
                v_b = all_vecs[word_to_idx[keys[1]]]
                v_c = all_vecs[word_to_idx[keys[2]]]
            except KeyError:
                questions_oov += 1
                categories[current_category]["oov"] += 1
                continue

            # 3COSADD query
            query = F.normalize((v_b - v_a + v_c).unsqueeze(0), dim=-1).squeeze(0)

            # Cosine similarities to all words
            sims = (all_vecs @ query).tolist()

            # Exclude the three input words
            exclude = {keys[0], keys[1], keys[2]}
            ranked  = sorted(
                [(all_words[i], sims[i])
                 for i in range(len(all_words))
                 if normalise(all_words[i]) not in exclude],
                key=lambda x: -x[1],
            )

            target_norm = keys[3]
            top_1_words = [normalise(ranked[0][0])] if ranked else []
            top_k_words = [normalise(w) for w, _ in ranked[:top_k]]

            if target_norm in top_1_words:
                correct_at_1 += 1
                categories[current_category]["correct_at_1"] += 1
            if target_norm in top_k_words:
                correct_at_k += 1
                categories[current_category]["correct_at_k"] += 1

    # Compute rates
    answered = questions_total - questions_oov
    acc_at_1 = correct_at_1 / max(answered, 1)
    acc_at_k = correct_at_k / max(answered, 1)

    results = {
        "total_questions":   questions_total,
        "oov_skipped":       questions_oov,
        "answered":          answered,
        "accuracy_at_1":     round(acc_at_1, 4),
        f"accuracy_at_{top_k}": round(acc_at_k, 4),
        "categories":        {},
    }

    for cat, stats in categories.items():
        ans = stats["total"] - stats["oov"]
        results["categories"][cat] = {
            "total":    stats["total"],
            "oov":      stats["oov"],
            "answered": ans,
            "acc_at_1": round(stats["correct_at_1"] / max(ans, 1), 4),
            f"acc_at_{top_k}": round(stats["correct_at_k"] / max(ans, 1), 4),
        }

    return results


# 2. Word similarity correlation
def evaluate_similarity(
    index:           EmbeddingIndex,
    similarity_path: str,
) -> dict:
    from scipy.stats import spearmanr

    model_sims  = []
    human_sims  = []
    oov_count   = 0
    total_pairs = 0

    with open(similarity_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\t,;]+|\s{2,}", line)
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue

            word_1, word_2 = parts[0], parts[1]
            try:
                human_score = float(parts[2])
            except ValueError:
                continue

            total_pairs += 1
            v1 = index.get(word_1)
            v2 = index.get(word_2)

            if v1 is None or v2 is None:
                oov_count += 1
                continue

            cos = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            model_sims.append(cos)
            human_sims.append(human_score)

    if len(model_sims) < 2:
        return {
            "error":       "Fewer than 2 word pairs found in index.",
            "total_pairs": total_pairs,
            "oov_pairs":   oov_count,
            "coverage":    0.0,
        }

    rho, pvalue = spearmanr(model_sims, human_sims)
    coverage    = len(model_sims) / max(total_pairs, 1)

    return {
        "spearman_rho":  round(float(rho),    4),
        "p_value":       round(float(pvalue),  6),
        "coverage":      round(coverage,       4),
        "pairs_scored":  len(model_sims),
        "oov_pairs":     oov_count,
        "total_pairs":   total_pairs,
    }


# Report printer
def print_analogy_report(results: dict, top_k: int) -> None:
    print(f"\n{'='*65}")
    print("  MORPHOLOGICAL ANALOGY TEST RESULTS")
    print(f"{'='*65}")
    print(f"  Total questions : {results['total_questions']:>6,}")
    print(f"  OOV skipped     : {results['oov_skipped']:>6,}")
    print(f"  Answered        : {results['answered']:>6,}")
    print(f"  Accuracy @1     : {results['accuracy_at_1']*100:>6.1f}%")
    print(f"  Accuracy @{top_k:<2}    : {results[f'accuracy_at_{top_k}']*100:>6.1f}%")

    if results["categories"]:
        print(f"\n  Per-category breakdown:")
        print(f"  {'Category':<30}  {'Answered':>8}  {'Acc@1':>7}  {'Acc@'+str(top_k):>7}")
        print(f"  {'-'*60}")
        for cat, stats in results["categories"].items():
            if stats["answered"] == 0:
                continue
            print(
                f"  {cat:<30}  {stats['answered']:>8,}  "
                f"{stats['acc_at_1']*100:>6.1f}%  "
                f"{stats[f'acc_at_{top_k}']*100:>6.1f}%"
            )
    print(f"{'='*65}")


def print_similarity_report(results: dict) -> None:
    print(f"\n{'='*65}")
    print("  WORD SIMILARITY CORRELATION RESULTS")
    print(f"{'='*65}")
    if "error" in results:
        print(f"  Error: {results['error']}")
    else:
        print(f"  Spearman rho    : {results['spearman_rho']:>8.4f}")
        print(f"  p-value         : {results['p_value']:>8.6f}")
        print(f"  Coverage        : {results['coverage']*100:>7.1f}%  "
              f"({results['pairs_scored']}/{results['total_pairs']} pairs)")
        print(f"  OOV pairs       : {results['oov_pairs']:>8,}")

        rho = results["spearman_rho"]
        if rho >= 0.60:
            level = "STRONG — embeddings capture genuine similarity well"
        elif rho >= 0.40:
            level = "MODERATE — meaningful correlation, room to improve"
        elif rho >= 0.20:
            level = "WEAK — some signal, needs more data or better model"
        else:
            level = "POOR — embeddings do not correlate with human judgement"
        print(f"\n  Interpretation  : {level}")
    print(f"{'='*65}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intrinsic evaluation for Arabic morphological embeddings."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to morpho_skipgram_best.pt"
    )
    parser.add_argument(
        "--vocab_dir", required=True,
        help="Path to vocabs/ directory"
    )
    parser.add_argument(
        "--word_list", default=None,
        help="Path to eval_words.jsonl (your existing eval word list)"
    )
    parser.add_argument(
        "--analogy", default=None,
        help="Path to analogy test file (4 words per line)"
    )
    parser.add_argument(
        "--similarity", default=None,
        help="Path to word similarity file (word1 word2 score per line)"
    )
    parser.add_argument(
        "--output_dir", default="eval_benchmarks/results",
        help="Directory to save JSON result files"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Accuracy@K for analogy test (default 5)"
    )
    args = parser.parse_args()

    if not args.analogy and not args.similarity:
        print("Provide at least --analogy or --similarity (or both).")
        raise SystemExit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build index
    index = EmbeddingIndex(
        checkpoint_path=args.checkpoint,
        vocab_dir=args.vocab_dir,
        word_list_path=args.word_list,
    )

    # Analogy evaluation
    if args.analogy:
        logger.info("Running analogy evaluation on %s ...", args.analogy)
        analogy_results = evaluate_analogies(
            index, args.analogy, top_k=args.top_k
        )
        print_analogy_report(analogy_results, args.top_k)

        out_path = out_dir / "analogy_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analogy_results, f, ensure_ascii=False, indent=2)
        logger.info("Analogy results saved -> %s", out_path)

    # Similarity evaluation
    if args.similarity:
        logger.info("Running similarity evaluation on %s ...", args.similarity)
        sim_results = evaluate_similarity(index, args.similarity)
        print_similarity_report(sim_results)

        out_path = out_dir / "similarity_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sim_results, f, ensure_ascii=False, indent=2)
        logger.info("Similarity results saved -> %s", out_path)

    print(f"\nAll results saved to: {out_dir}")
