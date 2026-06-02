from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class CachedSkipGramDataset(IterableDataset):
    def __init__(
        self,
        cache_path:  str,
        window_size: int  = 5,
        shuffle:     bool = True,
    ) -> None:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)

        self.tokens     = cache["tokens"]    # (N, 24)  LongTensor
        self.is_oov     = cache["is_oov"]    # (N,)     BoolTensor
        self.sent_ids   = cache["sent_ids"]  # (N,)     LongTensor
        self.word_to_id = cache["word_to_id"]
        self.meta       = cache["meta"]

        self.window_size        = window_size
        self.shuffle            = shuffle
        self.context_vocab_size = len(self.word_to_id)

        # Group token indices by sentence
        n_sents = int(self.sent_ids.max().item()) + 1
        self._sentences: list[list[int]] = [[] for _ in range(n_sents)]
        for idx, sid in enumerate(self.sent_ids.tolist()):
            self._sentences[sid].append(idx)

        logger.info(
            "Cache loaded: %d tokens | %d sentences | "
            "context vocab: %d | clean rate: %.1f%%",
            len(self.tokens),
            n_sents,
            self.context_vocab_size,
            self.meta.get("clean_rate", 0) * 100,
        )

    # Iteration
    def __iter__(self) -> Iterator[
        tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        sentence_order = list(range(len(self._sentences)))
        if self.shuffle:
            random.shuffle(sentence_order)

        for sent_idx in sentence_order:
            tok_indices = self._sentences[sent_idx]
            n = len(tok_indices)
            if n == 0:
                continue

            for pos_in_sent, center_idx in enumerate(tok_indices):
                row   = self.tokens[center_idx]          # (24,)
                oov   = self.is_oov[center_idx]          # bool

                morpho = {
                    "root":    row[0],
                    "pattern": row[1],
                    "affix":   row[2],
                    "pos":     row[3],
                }
                chars = row[4:24]    # cols 4-23 = 20 char indices (col 24 = word_id)

                # Context window (stays within sentence)
                left  = max(0, pos_in_sent - self.window_size)
                right = min(n - 1, pos_in_sent + self.window_size)

                for ctx_pos in range(left, right + 1):
                    if ctx_pos == pos_in_sent:
                        continue

                    # Context word is stored as surface word id in sent_ids
                    # We need the actual word — use the token's row index to
                    # look up through the word_to_id mapping stored at build time.
                    # Since we stored word ids implicitly via sent_ids ordering,
                    # we use the context token's root as a proxy for word lookup.
                    # Full word ids are stored as the last column if built with
                    # the updated build_cache — fall back to UNK if not present.
                    ctx_idx = tok_indices[ctx_pos]

                    # Context id: stored in col 24 if available, else UNK
                    if self.tokens.shape[1] > 24:
                        ctx_id = self.tokens[ctx_idx, 24]
                    else:
                        ctx_id = torch.tensor(1, dtype=torch.long)

                    yield morpho, chars, oov, ctx_id

    def __len__(self) -> int:
        """Estimated number of pairs."""
        n_tokens   = len(self.tokens)
        avg_ctx    = 2 * self.window_size
        return int(n_tokens * avg_ctx)


def skipgram_collate_fn(
    batch: list[tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    keys   = ["root", "pattern", "affix", "pos"]
    morpho = {k: torch.stack([b[0][k] for b in batch]) for k in keys}
    chars  = torch.stack([b[1] for b in batch])
    oov    = torch.stack([b[2] for b in batch]).bool()
    ctx    = torch.stack([b[3] for b in batch])
    return morpho, chars, oov, ctx