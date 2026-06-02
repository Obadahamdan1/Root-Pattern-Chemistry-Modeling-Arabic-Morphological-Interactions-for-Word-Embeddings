from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from morpho_dataset import StreamingSkipGramDataset, skipgram_collate_fn
from morpho_embedding_model import MorphoEmbeddingConfig, MorphoEmbeddingModel
from vocab_builder import VocabBuilder
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Trainer configuration
@dataclass
class TrainerConfig:
    data_path:        str   = "data/camel_2m.jsonl"
    vocab_dir:        str   = "vocabs/"
    output_dir:       str   = "checkpoints/"
    epochs:           int   = 3
    batch_size:       int   = 128
    lr:               float = 1e-3
    lr_div_factor:    float = 25.0
    final_div_factor: float = 1e4
    weight_decay:     float = 1e-5
    window_size:      int   = 5
    neg_samples:      int   = 5
    neg_sample_exp:   float = 0.75
    max_word_len:     int   = 20
    max_sentences:    int   = 500_000
    num_workers:      int   = 4
    log_every:        int   = 500
    save_every:       int   = 1
    device:           Optional[str] = None
    seed:             int   = 42
    model_cfg:        Optional[MorphoEmbeddingConfig] = None
    cache_path:       Optional[str] = None      # pre-built .pt cache from build_cache.py
    uniformity_weight: float = 0.1              # Wang & Isola uniformity loss weight


# Negative-sampling output layer
class SGNSOutputLayer(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.uniform_(self.embeddings.weight, -0.5 / embed_dim, 0.5 / embed_dim)

    def forward(
        self,
        center_vec: torch.Tensor,   # (B, D)
        pos_ids:    torch.Tensor,   # (B,)
        neg_ids:    torch.Tensor,   # (B, K)
    ) -> torch.Tensor:
        v_pos = self.embeddings(pos_ids)                                    # (B, D)
        v_neg = self.embeddings(neg_ids)                                    # (B, K, D)

        pos_score = (center_vec * v_pos).sum(dim=-1)                        # (B,)
        pos_loss  = F.logsigmoid(pos_score)

        neg_score = torch.bmm(v_neg, center_vec.unsqueeze(-1)).squeeze(-1)  # (B, K)
        neg_loss  = F.logsigmoid(-neg_score).sum(dim=-1)

        return -(pos_loss + neg_loss).mean()


# Negative sample generator

class NegativeSampler:
    """
    Draws negative samples from the unigram distribution raised to the
    power alpha (0.75), following Mikolov et al. (2013).
    """

    def __init__(
        self,
        word_frequencies: list[int],
        alpha:            float = 0.75,
        table_size:       int   = 10_000_000,
    ) -> None:
        freqs    = torch.tensor(word_frequencies, dtype=torch.float)
        freqs[0] = 0.0
        probs    = freqs.pow(alpha)
        probs   /= probs.sum()
        self._table = torch.multinomial(
            probs, table_size, replacement=True
        ).share_memory_()

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, len(self._table), (n,))
        return self._table[idx]

    def sample_batch(self, batch_size: int, k: int) -> torch.Tensor:
        return self.sample(batch_size * k).view(batch_size, k)


# Trainer

class SkipGramTrainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self._setup_device()
        self._setup_seed()
        self._setup_cpu_threads()

    # Setup
    def _setup_device(self) -> None:
        if self.cfg.device:
            self.device = torch.device(self.cfg.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info("Training device: %s", self.device)

    def _setup_seed(self) -> None:
        torch.manual_seed(self.cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.cfg.seed)

    def _setup_cpu_threads(self) -> None:
        if self.device.type == "cpu":
            n_cores = os.cpu_count() or 1
            torch.set_num_threads(n_cores)
            logger.info("CPU mode — using %d threads", n_cores)

    # Public API
    def train(self) -> None:
        """Full training pipeline with automatic checkpoint resuming."""

        vocab                             = self._load_vocab()
        loader, neg_sampler, context_size = self._build_dataloader(vocab)
        encoder, output_layer             = self._build_models(vocab, context_size)

        # Derive steps_per_epoch automatically from the dataset's __len__.
        # ds.__len__() = max_sentences * avg_tokens * 2*window_size.
        # This is accurate and requires no manual tuning.
        steps_per_epoch = math.ceil(len(loader.dataset) / self.cfg.batch_size)
        total_steps     = steps_per_epoch * self.cfg.epochs

        logger.info(
            "steps_per_epoch: %d  |  total_steps: %d  |  epochs: %d",
            steps_per_epoch, total_steps, self.cfg.epochs,
        )

        optimizer, scheduler = self._build_optimiser(
            encoder, output_layer, total_steps
        )

        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        best_loss = math.inf

        # Resume from checkpoint if available
        start_epoch = self._load_checkpoint(encoder, output_layer, optimizer)

        # Fast-forward scheduler to match resumed epoch.
        # Pure arithmetic — no tensors, no data. Takes ~15 seconds.
        if start_epoch > 1:
            steps_already_done = (start_epoch - 1) * steps_per_epoch
            logger.info(
                "Fast-forwarding scheduler by %d steps to match epoch %d...",
                steps_already_done, start_epoch,
            )
            for _ in range(steps_already_done):
                try:
                    scheduler.step()
                except ValueError:
                    break
            logger.info(
                "Scheduler ready. Current LR: %.2e",
                optimizer.param_groups[0]["lr"],
            )

        # Training loop
        for epoch in range(start_epoch, self.cfg.epochs + 1):
            epoch_loss = self._run_epoch(
                epoch, loader, encoder, output_layer,
                optimizer, scheduler, neg_sampler,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d — avg loss: %.4f  lr: %.2e",
                epoch, self.cfg.epochs, epoch_loss, current_lr,
            )

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self._save_checkpoint(
                    encoder, output_layer, optimizer, epoch,
                    epoch_loss, tag="best"
                )

            if self.cfg.save_every and epoch % self.cfg.save_every == 0:
                self._save_checkpoint(
                    encoder, output_layer, optimizer, epoch,
                    epoch_loss, tag=f"epoch{epoch:03d}"
                )

        logger.info("Training complete. Best loss: %.4f", best_loss)

    # Checkpoint resuming
    def _load_checkpoint(
        self,
        encoder:      MorphoEmbeddingModel,
        output_layer: SGNSOutputLayer,
        optimizer:    torch.optim.Optimizer,
    ) -> int:
 
        checkpoint_dir = Path(self.cfg.output_dir)
        checkpoints    = sorted(checkpoint_dir.glob("morpho_skipgram_epoch*.pt"))

        if not checkpoints:
            logger.info("No checkpoint found — starting fresh from epoch 1.")
            return 1

        latest = checkpoints[-1]
        logger.info("Found checkpoint: %s", latest)

        state = torch.load(latest, map_location=self.device)
        encoder.load_state_dict(state["encoder_state"])
        output_layer.load_state_dict(state["output_state"])
        optimizer.load_state_dict(state["optimizer_state"])

        completed = state["epoch"]
        logger.info(
            "Resumed — epoch %d completed. Starting epoch %d.",
            completed, completed + 1,
        )
        return completed + 1

    # Vocab loading
    def _load_vocab(self) -> VocabBuilder:
        return VocabBuilder.load(Path(self.cfg.vocab_dir))

    # DataLoader
    def _build_dataloader(
        self,
        vocab: VocabBuilder,
    ) -> tuple[DataLoader, NegativeSampler, int]:
        
        if self.cfg.cache_path:
            # Cache mode
            from cached_dataset import CachedSkipGramDataset
            from cached_dataset import skipgram_collate_fn as cache_collate
            ds = CachedSkipGramDataset(
                cache_path  = self.cfg.cache_path,
                window_size = self.cfg.window_size,
                shuffle     = True,
            )
            collate_fn   = cache_collate
            context_size = ds.context_vocab_size
            logger.info(
                "Cache mode | tokens: %d | context vocab: %d | "
                "estimated pairs: %d",
                len(ds.tokens), context_size, len(ds),
            )
        else:
            # Stream mode
            ds = StreamingSkipGramDataset(
                self.cfg.data_path,
                vocab,
                window_size   = self.cfg.window_size,
                max_word_len  = self.cfg.max_word_len,
                max_sentences = self.cfg.max_sentences,
            )
            collate_fn   = skipgram_collate_fn
            context_size = ds.context_vocab_size
            logger.info(
                "Stream mode | context vocab: %d | max_sentences: %d | "
                "estimated pairs: %d",
                context_size, self.cfg.max_sentences, len(ds),
            )

        # num_workers=0 in cache mode avoids pickling the large tensor
        n_workers = 0 if self.cfg.cache_path else self.cfg.num_workers

        loader = DataLoader(
            ds,
            batch_size        = self.cfg.batch_size,
            shuffle           = False,   # CachedSkipGramDataset shuffles internally
            num_workers       = n_workers,
            collate_fn        = collate_fn,
            pin_memory        = (self.device.type == "cuda"),
            drop_last         = True,
            persistent_workers= (n_workers > 0),
        )

        freq_list    = [1] * context_size
        freq_list[0] = 0
        neg_sampler  = NegativeSampler(freq_list, alpha=self.cfg.neg_sample_exp)

        return loader, neg_sampler, context_size

    # Model construction
    def _build_models(
        self,
        vocab:              VocabBuilder,
        context_vocab_size: int,
    ) -> tuple[MorphoEmbeddingModel, SGNSOutputLayer]:
        if self.cfg.model_cfg is not None:
            model_cfg = self.cfg.model_cfg
        else:
            model_cfg = MorphoEmbeddingConfig(**vocab.vocab_sizes)

        encoder      = MorphoEmbeddingModel(cfg=model_cfg).to(self.device)
        output_layer = SGNSOutputLayer(
            vocab_size=context_vocab_size,
            embed_dim=model_cfg.proj_dim,
        ).to(self.device)

        total_params = sum(
            p.numel() for p in
            list(encoder.parameters()) + list(output_layer.parameters())
            if p.requires_grad
        )
        logger.info("Total trainable parameters: %s", f"{total_params:,}")
        return encoder, output_layer

    # Optimiser + Scheduler
    def _build_optimiser(
        self,
        encoder:      MorphoEmbeddingModel,
        output_layer: SGNSOutputLayer,
        total_steps:  int,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.OneCycleLR]:
        params    = list(encoder.parameters()) + list(output_layer.parameters())
        optimizer = torch.optim.Adam(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.lr,
            total_steps=max(total_steps, 1),
            div_factor=self.cfg.lr_div_factor,
            final_div_factor=self.cfg.final_div_factor,
            anneal_strategy="cos",
            pct_start=0.3,
        )
        logger.info(
            "OneCycleLR | max_lr=%.2e | total_steps=%d | warmup_steps=%d",
            self.cfg.lr, total_steps, int(total_steps * 0.3),
        )
        return optimizer, scheduler

    # One epoch
    def _run_epoch(
        self,
        epoch:        int,
        loader:       DataLoader,
        encoder:      MorphoEmbeddingModel,
        output_layer: SGNSOutputLayer,
        optimizer:    torch.optim.Optimizer,
        scheduler:    torch.optim.lr_scheduler.OneCycleLR,
        neg_sampler:  NegativeSampler,
    ) -> float:
        encoder.train()
        output_layer.train()

        total_loss = 0.0
        n_batches  = 0
        t_start    = time.perf_counter()

        for step, (morpho, chars, oov, context_ids) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", leave=False,
                 total=math.ceil(len(loader.dataset) / self.cfg.batch_size)),
            1,
        ):
            morpho      = {k: v.to(self.device, non_blocking=True)
                           for k, v in morpho.items()}
            chars       = chars.to(self.device, non_blocking=True)
            oov         = oov.to(self.device, non_blocking=True).bool()
            context_ids = context_ids.to(self.device, non_blocking=True)

            B       = chars.size(0)
            neg_ids = neg_sampler.sample_batch(B, self.cfg.neg_samples).to(
                self.device, non_blocking=True
            )

            center_vec = encoder(morpho, chars, oov)           # (B, D) L2-normed

            # SGNS loss
            sgns_loss = output_layer(center_vec, context_ids, neg_ids)

            #  Uniformity loss (Wang & Isola 2020) 
            # Prevents embedding collapse by pushing vectors apart on the
            # unit sphere. Essential for this architecture — without it all
            # embeddings collapse to a single point (see ablation in paper).
            if self.cfg.uniformity_weight > 0.0 and B > 1:
                sq_dists = torch.pdist(center_vec, p=2).pow(2)
                u_loss   = sq_dists.mul(-2.0).exp().mean().log()
                loss     = sgns_loss + self.cfg.uniformity_weight * u_loss
            else:
                loss = sgns_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(output_layer.parameters()),
                max_norm=5.0,
            )
            optimizer.step()

            try:
                scheduler.step()
            except ValueError:
                pass

            total_loss += loss.item()
            n_batches  += 1

            if step % self.cfg.log_every == 0:
                elapsed  = time.perf_counter() - t_start
                avg_loss = total_loss / n_batches
                cur_lr   = optimizer.param_groups[0]["lr"]
                logger.info(
                    "  Epoch %d | step %6d | loss %.4f | lr %.2e | %.1f s",
                    epoch, step, avg_loss, cur_lr, elapsed,
                )

        return total_loss / max(n_batches, 1)

    # Checkpointing
    def _save_checkpoint(
        self,
        encoder:      MorphoEmbeddingModel,
        output_layer: SGNSOutputLayer,
        optimizer:    torch.optim.Optimizer,
        epoch:        int,
        loss:         float,
        tag:          str = "",
    ) -> None:
        name = f"morpho_skipgram_{tag}.pt" if tag else "morpho_skipgram.pt"
        path = Path(self.cfg.output_dir) / name
        torch.save(
            {
                "epoch":           epoch,
                "loss":            loss,
                "encoder_state":   encoder.state_dict(),
                "output_state":    output_layer.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_cfg":       encoder.cfg,
            },
            path,
        )
        logger.info("Checkpoint saved -> %s", path)

    # Helpers
    @staticmethod
    def _iter_jsonl(path: str):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


# CLI entry point

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Train Arabic Morphological Skip-gram embeddings."
    )
    parser.add_argument("--data_path",      default="data/camel_2m.jsonl")
    parser.add_argument("--vocab_dir",      default="vocabs/")
    parser.add_argument("--output_dir",     default="checkpoints/")
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--batch_size",     type=int,   default=128)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--window",         type=int,   default=5)
    parser.add_argument("--neg_samples",    type=int,   default=5)
    parser.add_argument("--max_word_len",   type=int,   default=20)
    parser.add_argument("--max_sentences",    type=int,   default=200_000)
    parser.add_argument("--num_workers",      type=int,   default=4)
    parser.add_argument("--log_every",        type=int,   default=500)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--cache_path",       default=None,
                        help="Pre-built cache .pt from build_cache.py. "
                             "Enables fast GPU training — skips JSON streaming.")
    parser.add_argument("--uniformity_weight",type=float, default=0.5,
                        help="Weight for uniformity loss. 0.0 = disabled.")
    args = parser.parse_args()

    cfg = TrainerConfig(
        data_path         = args.data_path,
        vocab_dir         = args.vocab_dir,
        output_dir        = args.output_dir,
        epochs            = args.epochs,
        batch_size        = args.batch_size,
        lr                = args.lr,
        window_size       = args.window,
        neg_samples       = args.neg_samples,
        max_word_len      = args.max_word_len,
        max_sentences     = args.max_sentences,
        num_workers       = args.num_workers,
        log_every         = args.log_every,
        seed              = args.seed,
        cache_path        = args.cache_path,
        uniformity_weight = args.uniformity_weight,
    )

    trainer = SkipGramTrainer(cfg)
    trainer.train()