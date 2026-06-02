from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Configuration dataclass

@dataclass
class MorphoEmbeddingConfig:
    root_vocab_size:    int   = 10_000
    pattern_vocab_size: int   =  2_000
    affix_vocab_size:   int   =  3_000
    pos_vocab_size:     int   =    100
    char_vocab_size:    int   =    150   # Arabic chars + diacritics + special tokens

    component_dim:      int   =    128
    proj_dim:           int   =    256

    char_embed_dim:     int   =    64
    char_kernel_sizes:  list  = field(default_factory=lambda: [2, 3, 4, 5])
    char_out_channels:  int   =    64   # filters per kernel width

    num_attn_heads:     int   =    4
    num_encoder_layers: int   =    2
    dropout:            float =    0.1

    max_word_len:       int   =   20
    padding_idx:        int   =    0



# Sub-modules

class ComponentEmbeddings(nn.Module):

    def __init__(self, cfg: MorphoEmbeddingConfig) -> None:
        super().__init__()
        dim = cfg.component_dim
        pad = cfg.padding_idx

        self.root_emb    = nn.Embedding(cfg.root_vocab_size,    dim, padding_idx=pad)
        self.pattern_emb = nn.Embedding(cfg.pattern_vocab_size, dim, padding_idx=pad)
        self.affix_emb   = nn.Embedding(cfg.affix_vocab_size,   dim, padding_idx=pad)
        self.pos_emb     = nn.Embedding(cfg.pos_vocab_size,     dim, padding_idx=pad)

        self.layer_norm  = nn.LayerNorm(dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for table in (self.root_emb, self.pattern_emb,
                      self.affix_emb, self.pos_emb):
            nn.init.normal_(table.weight, mean=0.0,
                            std=1.0 / math.sqrt(table.embedding_dim))

    def forward(
        self,
        root_idx:    torch.Tensor,   # (B,)
        pattern_idx: torch.Tensor,   # (B,)
        affix_idx:   torch.Tensor,   # (B,)
        pos_idx:     torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        v_root    = self.layer_norm(self.root_emb(root_idx))
        v_pattern = self.layer_norm(self.pattern_emb(pattern_idx))
        v_affix   = self.layer_norm(self.affix_emb(affix_idx))
        v_pos     = self.layer_norm(self.pos_emb(pos_idx))
        return v_root, v_pattern, v_affix, v_pos


class CharacterCNNEncoder(nn.Module):
   

    def __init__(self, cfg: MorphoEmbeddingConfig) -> None:
        super().__init__()
        self.char_embed = nn.Embedding(
            cfg.char_vocab_size, cfg.char_embed_dim, padding_idx=cfg.padding_idx
        )

        # One Conv1d per kernel width
        self.convolutions = nn.ModuleList([
            nn.Conv1d(
                in_channels=cfg.char_embed_dim,
                out_channels=cfg.char_out_channels,
                kernel_size=k,
                padding=0,
            )
            for k in cfg.char_kernel_sizes
        ])

        cnn_output_dim = len(cfg.char_kernel_sizes) * cfg.char_out_channels
        self.proj = nn.Linear(cnn_output_dim, cfg.proj_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, char_indices: torch.Tensor) -> torch.Tensor:

        # (B, L, E)  →  (B, E, L)  for Conv1d
        x = self.char_embed(char_indices).transpose(1, 2)

        pooled_features: list[torch.Tensor] = []
        for conv in self.convolutions:
            # (B, C_out, L - k + 1)
            activated = F.gelu(conv(x))
            # Global max pooling  →  (B, C_out)
            pooled = activated.max(dim=-1).values
            pooled_features.append(pooled)

        # (B, num_kernels * C_out)
        concat = torch.cat(pooled_features, dim=-1)
        return self.dropout(self.proj(concat))   # (B, proj_dim)


class MorphoTransformerFuser(nn.Module):
    
    def __init__(self, cfg: MorphoEmbeddingConfig) -> None:
        super().__init__()
        # Positional encoding for the 4-token component sequence
        self.register_buffer(
            "pos_enc",
            self._build_sinusoidal_pos_enc(seq_len=4, dim=cfg.component_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.component_dim,
            nhead=cfg.num_attn_heads,
            dim_feedforward=cfg.component_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,   # (B, S, E) convention
            norm_first=True,    # Pre-LN: more stable for small models
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_encoder_layers,
            enable_nested_tensor=False,
        )

        self.out_proj = nn.Linear(cfg.component_dim, cfg.proj_dim)
        self.dropout  = nn.Dropout(cfg.dropout)

    @staticmethod
    def _build_sinusoidal_pos_enc(seq_len: int, dim: int) -> torch.Tensor:
        pe  = torch.zeros(seq_len, dim)
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10_000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, seq_len, dim)

    def forward(
        self,
        v_root:    torch.Tensor,   # (B, D)
        v_pattern: torch.Tensor,   # (B, D)
        v_affix:   torch.Tensor,   # (B, D)
        v_pos:     torch.Tensor,   # (B, D)
    ) -> torch.Tensor:
        # Stack into sequence  →  (B, 4, component_dim)
        seq = torch.stack([v_root, v_pattern, v_affix, v_pos], dim=1)
        # Add positional encoding (broadcast over batch)
        seq = seq + self.pos_enc

        # Transformer Encoder  →  (B, 4, component_dim)
        encoded = self.transformer(seq)

        # Mean-pool over the 4 component positions  →  (B, component_dim)
        pooled = encoded.mean(dim=1)

        return self.dropout(self.out_proj(pooled))   # (B, proj_dim)



# Main model
class MorphoEmbeddingModel(nn.Module):
    def __init__(
        self,
        vocab_sizes: Optional[dict[str, int]] = None,
        cfg: Optional[MorphoEmbeddingConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or MorphoEmbeddingConfig()

        # Apply any vocab_size overrides from the caller
        if vocab_sizes:
            for key, size in vocab_sizes.items():
                attr = f"{key}_vocab_size"
                if hasattr(self.cfg, attr):
                    setattr(self.cfg, attr, size)

        c = self.cfg

        #Sub-modules
        self.component_embeds = ComponentEmbeddings(c)
        self.transformer_fuser = MorphoTransformerFuser(c)
        self.char_cnn = CharacterCNNEncoder(c)

        # Path (B): linear projection of raw concatenation
        #   v_proj = W · [v_root ; v_pattern ; v_affix ; v_pos] + b
        self.linear_proj = nn.Sequential(
            nn.Linear(c.component_dim * 4, c.proj_dim, bias=True),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )

        # Learnable scalar gate α ∈ (0,1): blends transformer vs linear path
        self.alpha_gate = nn.Parameter(torch.tensor(0.5))

        # Learnable scalar β: baseline CNN contribution (even when in-vocab)
        self.beta_base = nn.Parameter(torch.tensor(0.1))

        # Output normalisation
        self.output_norm = nn.LayerNorm(c.proj_dim)

    # Forward pass

    def forward(
        self,
        morpho_indices: dict[str, torch.LongTensor],
        char_indices:   torch.LongTensor,
        oov_mask:       Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        root_idx    = morpho_indices["root"]
        pattern_idx = morpho_indices["pattern"]
        affix_idx   = morpho_indices["affix"]
        pos_idx     = morpho_indices["pos"]

        # (1) Component embeddings
        v_root, v_pattern, v_affix, v_pos = self.component_embeds(
            root_idx, pattern_idx, affix_idx, pos_idx
        )

        # (2A) Attention-fused path
        v_attn = self.transformer_fuser(v_root, v_pattern, v_affix, v_pos)

        # (2B) Linear projection path
        concat   = torch.cat([v_root, v_pattern, v_affix, v_pos], dim=-1)
        v_linear = self.linear_proj(concat)

        # Blend the two morphological paths with learnable gate α
        alpha  = torch.sigmoid(self.alpha_gate)
        v_morph = alpha * v_attn + (1.0 - alpha) * v_linear   # (B, proj_dim)

        # (3) CNN character backoff
        v_char = self.char_cnn(char_indices)   # (B, proj_dim)

        # β_eff = β_base (always) +  (1 - β_base) when OOV (full override)
        beta = torch.sigmoid(self.beta_base)   # scalar, ∈ (0,1)
        if oov_mask is not None:
            # oov_mask: (B,) bool → float multiplier ∈ {β, 1.0}
            oov_weight = torch.where(
                oov_mask,
                torch.ones_like(oov_mask, dtype=torch.float),
                beta.expand_as(oov_mask.float()),
            )                                   # (B,)
            char_contribution = oov_weight.unsqueeze(-1) * v_char
        else:
            char_contribution = beta * v_char

        # (4) Interpolative fusion
        # v_final = (1-β)·v_morph + β·v_char  (convex combination)
        # Uses char_contribution which already accounts for OOV mask:
        #   - IV words:  weight = β      (CNN contributes moderately)
        #   - OOV words: weight = 1.0    (CNN takes full control)
        v_final = (1.0 - beta) * v_morph + char_contribution

        # (5) Output normalisation + L2 normalisation
        v_final = self.output_norm(v_final)
        v_final = F.normalize(v_final, p=2, dim=-1)

        return v_final   # (B, proj_dim)

    # Utility helpers
    def embed_word(
        self,
        morpho_indices: dict[str, torch.LongTensor],
        char_indices:   torch.LongTensor,
        oov_mask:       Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(morpho_indices, char_indices, oov_mask)

    def parameter_count(self) -> dict[str, int]:
        """Returns parameter counts broken down by sub-module."""
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "component_embeddings": count(self.component_embeds),
            "transformer_fuser":    count(self.transformer_fuser),
            "char_cnn":             count(self.char_cnn),
            "linear_proj":          count(self.linear_proj),
            "gates_and_norms":      (
                self.alpha_gate.numel() +
                self.beta_base.numel() +
                count(self.output_norm)
            ),
            "total":                count(self),
        }



# Quick smoke test

if __name__ == "__main__":
    torch.manual_seed(42)
    B = 4   # batch size

    cfg = MorphoEmbeddingConfig(
        root_vocab_size=10_000,
        pattern_vocab_size=2_000,
        affix_vocab_size=3_000,
        pos_vocab_size=100,
        char_vocab_size=150,
        component_dim=128,
        proj_dim=256,
        char_embed_dim=64,
        char_kernel_sizes=[2, 3, 4, 5],
        char_out_channels=64,
        num_attn_heads=4,
        num_encoder_layers=2,
        dropout=0.1,
        max_word_len=20,
    )

    model = MorphoEmbeddingModel(cfg=cfg)
    model.eval()

    # Dummy inputs
    morpho = {
        "root":    torch.randint(1, cfg.root_vocab_size,    (B,)),
        "pattern": torch.randint(1, cfg.pattern_vocab_size, (B,)),
        "affix":   torch.randint(1, cfg.affix_vocab_size,   (B,)),
        "pos":     torch.randint(1, cfg.pos_vocab_size,     (B,)),
    }
    chars    = torch.randint(1, cfg.char_vocab_size, (B, cfg.max_word_len))
    oov_mask = torch.tensor([False, True, False, True])   # 2nd and 4th words are OOV

    output = model(morpho, chars, oov_mask)

    print("Output shape :", output.shape)            # Expected: (4, 256)
    print("L2 norms     :", output.norm(dim=-1))     # Should all be ≈ 1.0
    print()
    print("Parameter breakdown:")
    for name, count in model.parameter_count().items():
        print(f"  {name:<25} {count:>10,}")