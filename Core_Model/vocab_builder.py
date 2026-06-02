from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)



# Constants

PAD_TOKEN = "<PAD>"   # index 0 — must match MorphoEmbeddingConfig.padding_idx
UNK_TOKEN = "<UNK>"   # index 1 — used at inference for unseen tokens

SPECIAL_TOKENS: list[str] = [PAD_TOKEN, UNK_TOKEN]

# CAMeL-Tools field names expected in each processed record
CAMEL_ROOT_FIELD    = "root"
CAMEL_PATTERN_FIELD = "pattern"
CAMEL_PREFIX_FIELD  = "prefix"
CAMEL_SUFFIX_FIELD  = "suffix"
CAMEL_POS_FIELD     = "pos"
CAMEL_WORD_FIELD    = "word"    # surface form — used for character vocabulary


# Single-track vocabulary

@dataclass
class Vocab:
   
    name:      str
    min_freq:  int = 1
    token2id:  dict[str, int] = field(default_factory=dict)
    id2token:  list[str]      = field(default_factory=list)

    # Build
    def build_from_counter(self, counter: Counter[str]) -> None:
        self.token2id = {}
        self.id2token = []

        # Reserve indices 0 and 1 for special tokens
        for special in SPECIAL_TOKENS:
            self._add_token(special)

        # Add tokens that meet the minimum frequency threshold,
        # sorted descending by frequency for deterministic ordering
        for token, freq in sorted(counter.items(), key=lambda x: -x[1]):
            if freq >= self.min_freq and token not in self.token2id:
                self._add_token(token)

        logger.info(
            "[%s] vocab built: %d tokens (%d special + %d content)",
            self.name,
            len(self.id2token),
            len(SPECIAL_TOKENS),
            len(self.id2token) - len(SPECIAL_TOKENS),
        )

    def _add_token(self, token: str) -> None:
        idx = len(self.id2token)
        self.token2id[token] = idx
        self.id2token.append(token)

    # Lookup helpers
    def encode(self, token: str) -> int:
        """Map a token string to its integer index (falls back to <UNK>=1)."""
        return self.token2id.get(token, self.token2id[UNK_TOKEN])

    def decode(self, idx: int) -> str:
        """Map an integer index back to its token string."""
        if idx < 0 or idx >= len(self.id2token):
            return UNK_TOKEN
        return self.id2token[idx]

    def __len__(self) -> int:
        return len(self.id2token)

    @property
    def pad_id(self) -> int:
        return self.token2id[PAD_TOKEN]   # always 0

    @property
    def unk_id(self) -> int:
        return self.token2id[UNK_TOKEN]   # always 1

    # Serialisation
    def to_dict(self) -> dict:
        return {
            "name":     self.name,
            "min_freq": self.min_freq,
            "id2token": self.id2token,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocab":
        obj = cls(name=data["name"], min_freq=data.get("min_freq", 1))
        obj.id2token = data["id2token"]
        obj.token2id = {tok: idx for idx, tok in enumerate(obj.id2token)}
        return obj


# VocabBuilder — orchestrates all tracks

class VocabBuilder:

    AFFIX_SEP = "+"

    def __init__(
        self,
        root_min_freq:    int = 2,
        pattern_min_freq: int = 2,
        affix_min_freq:   int = 1,
        pos_min_freq:     int = 1,
        char_min_freq:    int = 1,
        affix_sep:        str = "+",
    ) -> None:
        self.affix_sep = affix_sep

        self.root_vocab    = Vocab("root",    min_freq=root_min_freq)
        self.pattern_vocab = Vocab("pattern", min_freq=pattern_min_freq)
        self.affix_vocab   = Vocab("affix",   min_freq=affix_min_freq)
        self.pos_vocab     = Vocab("pos",     min_freq=pos_min_freq)
        self.char_vocab    = Vocab("char",    min_freq=char_min_freq)

        # Internal counters — populated during build_from_dataset
        self._root_counter:    Counter[str] = Counter()
        self._pattern_counter: Counter[str] = Counter()
        self._affix_counter:   Counter[str] = Counter()
        self._pos_counter:     Counter[str] = Counter()
        self._char_counter:    Counter[str] = Counter()

        self._built: bool = False

    # Build API
    def build_from_dataset(
        self,
        records: Iterator[dict],
        verbose: bool = True,
    ) -> None:
        """
        Modified to handle the nested 'tokens' structure from your previous script.
        """
        logger.info("Starting vocabulary scan …")
        n = 0

        for line_data in records:
            # New Step: Get the list of tokens from each line
            tokens = line_data.get("tokens", [])
            
            for record in tokens:
                # Map your specific keys to the ones expected by the new model
                # Old script uses 'lexeme_pattern', new one expects 'pattern'
                root    = record.get("root", "") or ""
                pattern = record.get("lexeme_pattern", record.get("surface_pattern", "")) or ""
                prefix_list  = record.get("prefixes", [])
                suffix_list  = record.get("suffixes", [])
                pos     = record.get("pos", "") or ""
                word    = record.get("word", record.get("text", "")) or ""

                # Handle the list-based prefixes/suffixes from your old data
                prefix = "".join(prefix_list) if isinstance(prefix_list, list) else prefix_list
                suffix = "".join(suffix_list) if isinstance(suffix_list, list) else suffix_list
                
                # Combine prefix + suffix into a single affix key for the new model
                affix = f"{prefix}{self.affix_sep}{suffix}"

                # Skip entirely empty records
                if not any([root, pattern, affix, pos, word]):
                    continue

                if root:    self._root_counter[root]       += 1
                if pattern: self._pattern_counter[pattern] += 1
                if affix:   self._affix_counter[affix]     += 1
                if pos:     self._pos_counter[pos]         += 1

                for ch in word:
                    self._char_counter[ch] += 1

                n += 1
            
            if verbose and n % 100_000 == 0:
                logger.info("  … scanned %d records", n)

        logger.info("Scan complete — %d records processed.", n)
        self._finalise()

    def _finalise(self) -> None:
        """Convert frequency counters into Vocab objects."""
        self.root_vocab.build_from_counter(self._root_counter)
        self.pattern_vocab.build_from_counter(self._pattern_counter)
        self.affix_vocab.build_from_counter(self._affix_counter)
        self.pos_vocab.build_from_counter(self._pos_counter)
        self.char_vocab.build_from_counter(self._char_counter)
        self._built = True
        logger.info(
            "Vocabulary sizes — root:%d  pattern:%d  affix:%d  pos:%d  char:%d",
            len(self.root_vocab),
            len(self.pattern_vocab),
            len(self.affix_vocab),
            len(self.pos_vocab),
            len(self.char_vocab),
        )

    # Encoding helpers

    def encode_record(self, record: dict) -> dict[str, int]:
        self._assert_built()

        root    = record.get(CAMEL_ROOT_FIELD,    "") or ""
        pattern = record.get(CAMEL_PATTERN_FIELD, "") or ""
        prefix  = record.get(CAMEL_PREFIX_FIELD,  "") or ""
        suffix  = record.get(CAMEL_SUFFIX_FIELD,  "") or ""
        pos     = record.get(CAMEL_POS_FIELD,     "") or ""
        affix   = f"{prefix}{self.affix_sep}{suffix}"

        # OOV if any critical field is absent or mapped to UNK
        is_oov = not root or not pattern or not pos

        return {
            "root":    self.root_vocab.encode(root),
            "pattern": self.pattern_vocab.encode(pattern),
            "affix":   self.affix_vocab.encode(affix),
            "pos":     self.pos_vocab.encode(pos),
            "is_oov":  is_oov,
        }

    def encode_chars(self, word: str, max_len: int = 20) -> list[int]:
        self._assert_built()
        ids = [self.char_vocab.encode(ch) for ch in word[:max_len]]
        # Right-pad with PAD index (0)
        ids += [self.char_vocab.pad_id] * (max_len - len(ids))
        return ids

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path) -> None:
        self._assert_built()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        tracks = {
            "root_vocab":    self.root_vocab,
            "pattern_vocab": self.pattern_vocab,
            "affix_vocab":   self.affix_vocab,
            "pos_vocab":     self.pos_vocab,
            "char_vocab":    self.char_vocab,
        }
        for filename, vocab in tracks.items():
            path = out / f"{filename}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(vocab.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info("Saved %s → %s", filename, path)

        # Metadata file for quick inspection / config wiring
        meta = {
            "built_at":    datetime.now(timezone.utc).isoformat(),
            "affix_sep":   self.affix_sep,
            "vocab_sizes": {name: len(v) for name, v in tracks.items()},
        }
        meta_path = out / "vocab_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("Saved vocab_meta.json → %s", meta_path)

    @classmethod
    def load(cls, vocab_dir: str | Path) -> "VocabBuilder":
        d = Path(vocab_dir)
        obj = cls.__new__(cls)
        obj._built    = True
        obj.affix_sep = cls.AFFIX_SEP

        # Load optional metadata for affix separator override
        meta_path = d / "vocab_meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            obj.affix_sep = meta.get("affix_sep", cls.AFFIX_SEP)

        track_map = {
            "root_vocab":    "root_vocab",
            "pattern_vocab": "pattern_vocab",
            "affix_vocab":   "affix_vocab",
            "pos_vocab":     "pos_vocab",
            "char_vocab":    "char_vocab",
        }
        for filename, attr in track_map.items():
            path = d / f"{filename}.json"
            with open(path, encoding="utf-8") as f:
                setattr(obj, attr, Vocab.from_dict(json.load(f)))

        # Silence counter attributes — not needed post-load
        obj._root_counter    = Counter()
        obj._pattern_counter = Counter()
        obj._affix_counter   = Counter()
        obj._pos_counter     = Counter()
        obj._char_counter    = Counter()

        logger.info("VocabBuilder loaded from %s", d)
        return obj

    # Internal helpers
    def _assert_built(self) -> None:
        if not self._built:
            raise RuntimeError(
                "VocabBuilder has not been built yet. "
                "Call build_from_dataset() or load() first."
            )

    # Properties (for convenient wiring into MorphoEmbeddingConfig)
    @property
    def vocab_sizes(self) -> dict[str, int]:
        self._assert_built()
        return {
            "root_vocab_size":    len(self.root_vocab),
            "pattern_vocab_size": len(self.pattern_vocab),
            "affix_vocab_size":   len(self.affix_vocab),
            "pos_vocab_size":     len(self.pos_vocab),
            "char_vocab_size":    len(self.char_vocab),
        }


# CLI entry point

def _iter_jsonl(path: str) -> Iterator[dict]:
    """Yield records from a newline-delimited JSON file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Build morphological vocabularies from a CAMeL-processed dataset."
    )
    parser.add_argument("--data_path",  required=True,
                        help="Path to processed .jsonl file.")
    parser.add_argument("--output_dir", default="vocabs/",
                        help="Directory to save vocabulary JSON files.")
    parser.add_argument("--root_min_freq",    type=int, default=2)
    parser.add_argument("--pattern_min_freq", type=int, default=2)
    parser.add_argument("--affix_min_freq",   type=int, default=1)
    parser.add_argument("--pos_min_freq",     type=int, default=1)
    parser.add_argument("--char_min_freq",    type=int, default=1)
    args = parser.parse_args()

    builder = VocabBuilder(
        root_min_freq=args.root_min_freq,
        pattern_min_freq=args.pattern_min_freq,
        affix_min_freq=args.affix_min_freq,
        pos_min_freq=args.pos_min_freq,
        char_min_freq=args.char_min_freq,
    )
    builder.build_from_dataset(_iter_jsonl(args.data_path))
    builder.save(args.output_dir)
    print("\nVocab sizes:")
    for k, v in builder.vocab_sizes.items():
        print(f"  {k:<25} {v:>8,}")
