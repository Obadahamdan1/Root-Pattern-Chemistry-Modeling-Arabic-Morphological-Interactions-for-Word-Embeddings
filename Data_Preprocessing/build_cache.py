from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Quality filter (same logic as morpho_dataset.py  must stay in sync)
def _strip_root(root: str) -> str:
    return root.replace(".", "").replace("-", "").strip()


def is_clean(raw: dict) -> bool:
    root_raw = raw.get("root", "") or ""
    pattern  = (raw.get("lexeme_pattern")
                or raw.get("surface_pattern")
                or raw.get("pattern", "")) or ""
    pos      = raw.get("pos", "") or ""
    root     = _strip_root(root_raw)

    if not root or not pattern or not pos:
        return False
    # NTWS before length check
    if root.upper() in ("NTWS", "FOREIGN", "NO_ANALYSIS"):
        return False
    if pattern.upper() in ("NTWS", "FOREIGN", "NO_ANALYSIS"):
        return False
    if len(root) not in (3, 4):
        return False
    if "#" in root or "#" in root_raw or "#" in pattern:
        return False
    if pos in ("punc", "digit"):
        return False
    return True


def normalise_record(raw: dict) -> dict:
    word = raw.get("word", raw.get("surface", "")) or ""
    return {
        "word":    word,
        "root":    raw.get("root", "") or "",
        "pattern": (raw.get("lexeme_pattern")
                    or raw.get("surface_pattern")
                    or raw.get("pattern", "") or ""),
        "prefix":  "".join(raw.get("prefixes", [])),
        "suffix":  "".join(raw.get("suffixes", [])),
        "pos":     raw.get("pos", "") or "",
    }


# Surface word vocab builder
def build_word_vocab(
    data_path:     str,
    max_sentences: int,
    min_freq:      int = 5,
) -> dict[str, int]:
    logger.info("Building surface word vocabulary (scan up to %d sents)...",
                max_sentences)
    counts: Counter = Counter()
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_sentences:
                break
            try:
                for tok in json.loads(line).get("tokens", []):
                    w = tok.get("word", tok.get("surface", "")) or ""
                    if w:
                        counts[w] += 1
            except json.JSONDecodeError:
                continue

    w2id = {"<PAD>": 0, "<UNK>": 1}
    for w, c in counts.items():
        if c >= min_freq:
            w2id[w] = len(w2id)

    logger.info("Surface vocab: %d words (freq >= %d)", len(w2id), min_freq)
    return w2id


# Main cache builder
def build_cache(
    data_path:     str,
    vocab_dir:     str,
    output_path:   str,
    max_sentences: int,
    window_size:   int = 5,
    max_word_len:  int = 20,
) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from vocab_builder import VocabBuilder

    t0 = time.perf_counter()

    # Load morphological vocab
    vocab = VocabBuilder.load(Path(vocab_dir))
    logger.info("Morphological vocab loaded from %s", vocab_dir)

    # Build surface word vocab
    word_to_id = build_word_vocab(data_path, max_sentences)

    # Storage lists — we collect everything then stack
    # Each token: [root_id, pattern_id, affix_id, pos_id, char_ids[20]] = 24 ints
    token_rows  = []   # list of list[int] length 24
    oov_flags   = []   # bool per token
    sent_ids    = []   # int per token — sentence index

    sentences_seen  = 0
    tokens_seen     = 0
    tokens_kept     = 0
    tokens_filtered = 0

    logger.info("Encoding corpus (max %d sentences)...", max_sentences)
    log_every = max(1, max_sentences // 20)   # log ~20 times total

    with open(data_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if sentences_seen >= max_sentences:
                break
            try:
                obj      = json.loads(line)
                sentence = obj.get("tokens", [])
                if not sentence:
                    continue

                sentences_seen += 1
                sent_tok_indices = []  # indices into token_rows for this sentence

                for tok_raw in sentence:
                    tokens_seen += 1

                    if not is_clean(tok_raw):
                        tokens_filtered += 1
                        sent_tok_indices.append(None)
                        continue

                    rec  = normalise_record(tok_raw)
                    enc  = vocab.encode_record(rec)
                    char = vocab.encode_chars(rec["word"], max_len=max_word_len)

                    row = [
                        enc["root"],
                        enc["pattern"],
                        enc["affix"],
                        enc["pos"],
                        *char,                                    # cols 4-23
                        word_to_id.get(rec["word"], 1),           # col 24 = word id
                    ]

                    token_rows.append(row)
                    oov_flags.append(bool(enc["is_oov"]))
                    sent_ids.append(sentences_seen - 1)
                    sent_tok_indices.append(len(token_rows) - 1)
                    tokens_kept += 1

                if sentences_seen % log_every == 0:
                    elapsed = time.perf_counter() - t0
                    rate    = sentences_seen / elapsed
                    eta     = (max_sentences - sentences_seen) / max(rate, 1)
                    clean_pct = 100 * tokens_kept / max(tokens_seen, 1)
                    logger.info(
                        "  %6d / %d sents | %d tokens kept | "
                        "%.1f%% clean | %.0f sents/s | ETA %.0fm",
                        sentences_seen, max_sentences,
                        tokens_kept, clean_pct, rate, eta / 60,
                    )

            except json.JSONDecodeError:
                continue

    clean_rate = tokens_kept / max(tokens_seen, 1)
    logger.info(
        "Encoding done: %d sents | %d tokens seen | "
        "%d kept (%.1f%% clean) | %.1f min",
        sentences_seen, tokens_seen, tokens_kept,
        100 * clean_rate, (time.perf_counter() - t0) / 60,
    )

    # Stack into tensors
    logger.info("Stacking tensors...")
    tokens_tensor  = torch.tensor(token_rows, dtype=torch.long)   # (N, 24)
    oov_tensor     = torch.tensor(oov_flags,  dtype=torch.bool)   # (N,)
    sent_id_tensor = torch.tensor(sent_ids,   dtype=torch.long)   # (N,)

    cache = {
        "tokens":     tokens_tensor,
        "is_oov":     oov_tensor,
        "sent_ids":   sent_id_tensor,
        "word_to_id": word_to_id,
        "meta": {
            "n_sentences":  sentences_seen,
            "n_tokens":     tokens_kept,
            "window_size":  window_size,
            "max_word_len": max_word_len,
            "vocab_dir":    str(vocab_dir),
            "data_path":    str(data_path),
            "clean_rate":   round(clean_rate, 4),
        },
    }

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving cache to %s ...", output_path)
    torch.save(cache, output_path)

    size_mb = out.stat().st_size / 1024 / 1024
    total   = time.perf_counter() - t0
    logger.info("Cache saved: %.1f MB | Total time: %.1f min", size_mb, total / 60)
    logger.info("Tokens in cache: %d", tokens_kept)
    logger.info("Tensor shape: %s", tuple(tokens_tensor.shape))
    logger.info(
        "\nTo use: pass --cache_path \"%s\" to train_skipgram.py", output_path
    )


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build preprocessing cache for fast GPU training."
    )
    parser.add_argument("--data_path",     required=True,
                        help="Path to camel_2m.jsonl")
    parser.add_argument("--vocab_dir",     required=True,
                        help="Path to vocabs/ directory")
    parser.add_argument("--output",        required=True,
                        help="Output path for cache .pt file")
    parser.add_argument("--max_sentences", type=int, default=500_000)
    parser.add_argument("--window_size",   type=int, default=5)
    parser.add_argument("--max_word_len",  type=int, default=20)
    args = parser.parse_args()

    build_cache(
        data_path     = args.data_path,
        vocab_dir     = args.vocab_dir,
        output_path   = args.output,
        max_sentences = args.max_sentences,
        window_size   = args.window_size,
        max_word_len  = args.max_word_len,
    )