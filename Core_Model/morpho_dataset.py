from __future__ import annotations

import json
import logging
from typing import Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

from vocab_builder import VocabBuilder

logger = logging.getLogger(__name__)
Record = dict[str, str]


# Word-level dataset
class ArabicMorphoDataset(Dataset):
    """Maps a list of CAMeL-Tools word records to model-ready tensors."""

    def __init__(
        self,
        records:      list[Record],
        vocab:        VocabBuilder,
        max_word_len: int = 20,
    ) -> None:
        self.records      = records
        self.vocab        = vocab
        self.max_word_len = max_word_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        word   = record.get("word", "") or ""
        enc    = self.vocab.encode_record(record)
        morpho = {
            k: torch.tensor(enc[k], dtype=torch.long)
            for k in ("root", "pattern", "affix", "pos")
        }
        chars = torch.tensor(
            self.vocab.encode_chars(word, max_len=self.max_word_len),
            dtype=torch.long,
        )
        oov = torch.tensor(bool(enc["is_oov"]), dtype=torch.bool)
        return morpho, chars, oov


# Streaming skip-gram dataset
class StreamingSkipGramDataset(IterableDataset):

    MAX_VOCAB_SCAN: int   = 500_000
    CLEAN_RATE:     float = 0.57    # measured from camel_2m.jsonl quality audit

    def __init__(
        self,
        data_path:     str,
        vocab:         VocabBuilder,
        window_size:   int = 5,
        max_word_len:  int = 20,
        max_sentences: int = 200_000,
    ) -> None:
        self.data_path     = data_path
        self.vocab         = vocab
        self.window_size   = window_size
        self.max_word_len  = max_word_len
        self.max_sentences = max_sentences

        self.word_to_id         = self._build_word_vocab()
        self.context_vocab_size = len(self.word_to_id)

    # Surface-word vocabulary
    def _build_word_vocab(self) -> dict[str, int]:
        """Scan first MAX_VOCAB_SCAN lines to build surface-word id map."""
        from collections import Counter
        print(f"Building surface-word vocabulary "
              f"(scanning up to {self.MAX_VOCAB_SCAN:,} sentences)...")
        counts: Counter = Counter()

        with open(self.data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= self.MAX_VOCAB_SCAN:
                    break
                try:
                    for rec in json.loads(line).get("tokens", []):
                        w = rec.get("word", rec.get("surface", "")) or ""
                        if w:
                            counts[w] += 1
                except json.JSONDecodeError:
                    continue

        word_to_id: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        for w, c in counts.items():
            if c >= 5:
                word_to_id[w] = len(word_to_id)

        print(f"Surface vocab size: {len(word_to_id):,}  "
              f"(words with freq >= 5 in first {self.MAX_VOCAB_SCAN:,} sents)")
        return word_to_id

    # Parse quality filter
    @staticmethod
    def _is_clean_record(raw: dict) -> bool:
        root_raw = raw.get("root", "") or ""
        pattern  = (raw.get("lexeme_pattern")
                    or raw.get("surface_pattern")
                    or raw.get("pattern", "")) or ""
        pos      = raw.get("pos", "") or ""

        # Strip CAMeL dot separators before ALL checks
        root = root_raw.replace(".", "").replace("-", "").strip()

        # Must have all three fields
        if not root or not pattern or not pos:
            return False

        # FIX B: NTWS/FOREIGN checked BEFORE length (length 4 would pass otherwise)
        if root.upper() in ("NTWS", "FOREIGN", "NO_ANALYSIS"):
            return False
        if pattern.upper() in ("NTWS", "FOREIGN", "NO_ANALYSIS"):
            return False

        # FIX A: length check on dot-stripped root
        if len(root) not in (3, 4):
            return False

        # No # placeholders (partial parse) in either form
        if "#" in root or "#" in root_raw or "#" in pattern:
            return False

        # Punctuation and digits carry no morphological information
        if pos in ("punc", "digit"):
            return False

        return True

    # CAMeL field normalisation
    @staticmethod
    def _normalise_record(raw: dict) -> dict:
        """Map CAMeL-Tools field names to VocabBuilder.encode_record() schema."""
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

    # Iteration
    def __iter__(self) -> Iterator[
        tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Stream (center, context) pairs one sentence at a time.
        Opens a fresh file handle per epoch (epoch-safe).
        Skips center tokens that fail _is_clean_record.
        Stops cleanly after max_sentences.
        """
        sentences_seen  = 0
        tokens_seen     = 0
        tokens_filtered = 0

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):

                if self.max_sentences and sentences_seen >= self.max_sentences:
                    break

                try:
                    obj      = json.loads(line)
                    sentence = obj.get("tokens", [])
                    n        = len(sentence)

                    if n == 0:
                        continue

                    sentences_seen += 1

                    for i, center_raw in enumerate(sentence):
                        tokens_seen += 1

                        if not self._is_clean_record(center_raw):
                            tokens_filtered += 1
                            continue

                        cr   = self._normalise_record(center_raw)
                        enc  = self.vocab.encode_record(cr)
                        cids = self.vocab.encode_chars(cr["word"], self.max_word_len)

                        morpho = {
                            k: torch.tensor(enc[k], dtype=torch.long)
                            for k in ("root", "pattern", "affix", "pos")
                        }
                        chars = torch.tensor(cids, dtype=torch.long)
                        oov   = torch.tensor(bool(enc["is_oov"]), dtype=torch.bool)

                        left  = max(0, i - self.window_size)
                        right = min(n - 1, i + self.window_size)

                        for j in range(left, right + 1):
                            if j == i:
                                continue
                            ctx_word = (
                                sentence[j].get("word",
                                sentence[j].get("surface", "")) or ""
                            )
                            yield (
                                morpho,
                                chars,
                                oov,
                                torch.tensor(
                                    self.word_to_id.get(ctx_word, 1),
                                    dtype=torch.long,
                                ),
                            )

                except json.JSONDecodeError:
                    logger.warning("Malformed JSON at line %d", line_no)
                    continue

        clean_pct = 100 * (1 - tokens_filtered / max(tokens_seen, 1))
        logger.info(
            "Epoch done — %d sentences | %d tokens | %.1f%% clean",
            sentences_seen, tokens_seen, clean_pct,
        )

    def __len__(self) -> int:

        cap = self.max_sentences if self.max_sentences else 2_000_000
        return int(cap * 40 * 2 * self.window_size * self.CLEAN_RATE)


# Collate function
def skipgram_collate_fn(
    batch: list[tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    keys   = ["root", "pattern", "affix", "pos"]
    morpho = {k: torch.stack([b[0][k] for b in batch]) for k in keys}
    chars  = torch.stack([b[1] for b in batch])
    oov    = torch.stack([b[2] for b in batch]).bool()
    ctx    = torch.stack([b[3] for b in batch])
    return morpho, chars, oov, ctx


# Smoke test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("Testing _is_clean_record (Fix A: dot-strip, Fix B: NTWS before length)\n")
    tests = [
        # Dotted roots — all should PASS after dot-strip (Fix A)
        ({"root": "ك.ت.ب",   "pattern": "فَعَلَ",    "pos": "verb"}, True,  "ك.ت.ب dotted"),
        ({"root": "ع.د.د",   "pattern": "إِفْعال",   "pos": "noun"}, True,  "ع.د.د dotted"),
        ({"root": "د.ح.ر.ج", "pattern": "فَعْلَلَ",  "pos": "verb"}, True,  "4-letter dotted"),
        ({"root": "م.ن",     "pattern": "فِعْل",     "pos": "prep"}, False, "2-letter after strip"),
        # NTWS — must FAIL even though length 4 (Fix B)
        ({"root": "NTWS",    "pattern": "NTWS",       "pos": "adj"},  False, "NTWS length=4"),
        ({"root": "NTWS",    "pattern": "فَعَلَ",    "pos": "noun"}, False, "NTWS root"),
        # Hash failures
        ({"root": "ق.#.ل",   "pattern": "فَعَلَ",    "pos": "verb"}, False, "hash in dotted root"),
        ({"root": "كتب",     "pattern": "فَعَلَ",    "pos": "punc"}, False, "punc pos"),
        ({"root": "",        "pattern": "",            "pos": ""},     False, "all empty"),
        # Valid undotted roots (from tashkeela_analyzed.jsonl)
        ({"root": "كتب",     "pattern": "فَعَلَ",    "pos": "verb"}, True,  "undotted clean"),
        ({"root": "علم",     "pattern": "فَعْل",     "pos": "noun"}, True,  "undotted clean 2"),
    ]
    all_pass = True
    for raw, exp, label in tests:
        got    = StreamingSkipGramDataset._is_clean_record(raw)
        status = "PASS" if got == exp else "FAIL"
        if got != exp:
            all_pass = False
        print(f"  {status}  {label:<35}  expected={exp}  got={got}")

    print(f"\nAll tests passed: {all_pass}")

    print("\n__len__ estimates (CLEAN_RATE=0.57, Wikipedia):")
    class _FakeDS(StreamingSkipGramDataset):
        def _build_word_vocab(self):
            return {"<PAD>": 0, "<UNK>": 1}
    fake            = _FakeDS.__new__(_FakeDS)
    fake.window_size = 5
    for sents in [25_000, 200_000]:
        fake.max_sentences = sents
        pairs = fake.__len__()
        steps = pairs // 128
        hours = steps / 4 / 3600
        print(f"  {sents:>7,} sents → {pairs:>12,} pairs "
              f"→ {steps:>8,} steps → ~{hours:.1f}h per epoch")