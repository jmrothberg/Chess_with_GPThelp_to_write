"""
ChessBrain - Chess Move Prediction Transformer Model

A specialized transformer model for chess move prediction, stripped down from Brain6
for focused chess-only functionality. Uses chess move tokenization and game boundary masking.

BLACKWELL GB10 GPU TRAINING OPTIMIZATION
=======================================

THIS RAN ON DGX SPARX WITHOUT MEMROY LEAK ISSUES BUT NOW I can get the configuration to work again


Your Blackwell GB10 GPU (128GB unified memory) supports high-performance chess training:

1. SINGLE GPU TRAINING (Recommended for stability)
   - Optimized for Blackwell architecture with 128GB unified memory
   - Batch size 32 recommended for stability (higher sizes may cause kernel issues)
   - Maximum performance with Blackwell's advanced compute capabilities

2. MEMORY OPTIMIZATION
   - 128GB unified memory enables large model training
   - Conservative batch sizing prevents Blackwell kernel compatibility issues
   - PyTorch 2.5.1 + CUDA 12.4 provides stable Blackwell support

ARCHITECTURE OVERVIEW:
- Chess-specific tokenization (moves + game boundaries)
- MultiQueryAttention for efficient attention computation
- RMSNorm for improved training stability
- SwiGLU activation in feed-forward layers
- Game boundary masking for proper sequence separation

CURRENT STATUS (PyTorch 2.10.0.dev CUDA 13.0 nightly):
- ✅ Blackwell GB10 GPU: 128GB unified memory, compute capability 12.1
- ✅ Single GPU training: Optimized for Blackwell architecture
- ✅ Batch sizes: 64, 256, 1024+ all working (CUDA 13 kernel support)
- ✅ CUDA 13.0: Full Blackwell compatibility achieved
- ✅ Model architecture: Chess-optimized transformer with MultiQueryAttention

RECOMMENDED USAGE:
- For reliable training: Single GPU (GPU 0) with large batch sizes
- For faster training: Multi-GPU DataParallel (monitor for stability)
- Chess-specific optimizations: MultiQueryAttention, game masking, RMSNorm

CURRENT WORKING SETUP (DO NOT CHANGE):
====================================
- PyTorch: 2.9.0+cu128 (CUDA 12.8)
- System CUDA: 12.8
- GPU: NVIDIA GB10 Blackwell (128GB unified memory, compute capability 12.1)
- Batch sizes: 64, 256, 1024+ all working
- Save path: /home/jonathan/Data/ (not /data/)

⚠️  CRITICAL BLACKWELL MEMORY LEAK FIX (October 2025):
=====================================================
PyTorch 2.9+ DEPRECATED old environment variable names. Must use NEW names!

REQUIRED environment variables (set in .venv/bin/activate):
  export PYTORCH_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.8"
  export CUDA_DEVICE_MAX_CONNECTIONS=32
  export CUDA_AUTO_BOOST=0

WHY THIS IS CRITICAL:
- OLD variable: PYTORCH_CUDA_ALLOC_CONF (deprecated, PyTorch 2.9+ ignores it)
- NEW variable: PYTORCH_ALLOC_CONF (required for PyTorch 2.9+)
- Without these settings, Blackwell GPUs leak memory due to fragmentation
- Memory accumulates and eventually causes OOM crashes

IF YOU RECREATE .venv:
1. Run: ./setup_blackwell.sh (automatically adds variables to activation script)
2. OR manually add the exports above to .venv/bin/activate
3. Deactivate and reactivate venv to apply

TO VERIFY VARIABLES ARE SET:
  python3 -c "import os; print('PYTORCH_ALLOC_CONF:', os.getenv('PYTORCH_ALLOC_CONF'))"
  Should print the configuration, NOT None
"""

import os
import platform
import re
import torch
import torch.nn as nn
import torch.optim as optim
import math
import signal
import sys
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from transformers.optimization import Adafactor
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Prioritize MPS on Mac systems for native GPU support
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    device = torch.device('mps')
    gpu_indices = []  # MPS doesn't use gpu_indices like CUDA
    print("Using MPS GPU")
elif torch.cuda.is_available() and not os.environ.get('_CHESS_DDP_WORKER'):
    # Main process only — DDP workers skip this to avoid creating parasitic CUDA contexts on GPU 0
    gpu_indices = None

    os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

    # Train faster by allowing TF32 precision on A100 and newer GPUs if available
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True

    # Reduce memory fragmentation
    torch.cuda.empty_cache()

    # Better asynchronous GPU operations
    torch.cuda.set_stream(torch.cuda.Stream())

    # Force garbage collection to reduce memory fragmentation
    import gc
    gc.collect()

    device = torch.device('cuda')
    print("Using CUDA with optimized settings")
    # Set CUDA to release memory when possible - helps prevent OOM errors
    torch.cuda.empty_cache()
    # Conservative memory allocation to prevent crashes with VNC/Cinnamon
    torch.cuda.set_per_process_memory_fraction(0.80)
elif torch.cuda.is_available():
    # DDP worker process — minimal setup, per-rank device configured in _ddp_train_worker
    gpu_indices = None
    device = torch.device('cuda')
else:
    print("ERROR: No GPU available. This chess training requires GPU support (CUDA or MPS).")
    exit(1)

# Ctrl+C is handled by the training loop (train_chess_model) to allow
# switching data files without quitting. No signal handler needed.

# At the beginning of your script, after device selection:
if device.type == 'mps':
    torch.set_default_dtype(torch.float32)
    # Temporarily disable aggressive MPS optimizations to test GUI
    # torch.backends.mps.enable_ddp = True  # Enable distributed data parallel support
    # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Allow full memory usage
    # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable CPU fallback for unsupported ops
    print("🔥 Mac Studio MPS basic setup (aggressive optimizations temporarily disabled)")

# Define special tokens for chess games
special_tokens = ['<STARTGAME>', '<EOFG>']

# === Role-specific 4-token-per-ply grammar constants ===
# Each chess ply emits 4 tokens: COLOR, FROM, TO, PROMO
# Each token type gets its own output head, reducing vocab from ~20K to 140.
ROLE_COLOR = 0
ROLE_FROM = 1
ROLE_TO = 2
ROLE_PROMO = 3
ROLE_SPECIAL = -1

# Token offsets in the global 140-token vocabulary
COLOR_OFFSET = 0        # tokens 0..1   (White=0, Black=1)
FROM_OFFSET = 2         # tokens 2..65  (64 squares)
TO_OFFSET = 66          # tokens 66..129 (64 squares)
PROMO_OFFSET = 130      # tokens 130..134 (none, q, r, b, n)

# Special tokens
STARTGAME = 135
EOFG = 136
PAD = 137
W_RESULT = 138
D_RESULT = 139

ROLE_VOCAB_SIZE = 140

# Chess defaults - optimized for 10M Stockfish games on 4×48GB GPUs
CHESS_DEFAULTS = {
    'n_embd': 512,       # Embedding dimension (512/8=64 per head)
    'n_head': 8,         # Query heads
    'n_kv_heads': 2,     # KV heads (4:1 GQA ratio)
    'block_size': 512,   # 4-token mode default (512/4=128 half-moves=64 full moves)
    'n_layer': 12,       # Transformer layers - deeper for tactical depth
    'dropout': 0.0,      # No dropout - Stockfish games are deterministic, no noise to regularize
    'batch_size': 512,   # Split across 4 GPUs (128 per GPU) - safe for 48GB GPUs
    'num_epochs': 3,     # Short training sessions (1-3 epochs) - matches session-based training
    'learning_rate': 4e-4,  # Learning rate - stable for short sessions
    'weight_decay': 0.01,   # Standard regularization
    'max_norm': 5.0,     # Gradient clipping threshold (increased to reduce clipping frequency)
}

# Core model components for chess move prediction
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm provides an efficient alternative to standard LayerNorm by normalizing
    only by the root mean square of the features, without centering (mean subtraction).
    This reduces computation while maintaining training stability.

    Key advantages over LayerNorm:
    - Faster computation (no mean calculation)
    - Better gradient flow in deep networks
    - Equivalent performance to LayerNorm in practice

    Args:
        dim: Feature dimension to normalize
        eps: Small epsilon for numerical stability (default: 1e-5)
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute RMS normalization: x / sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight


class FeedForward(nn.Module):
    """
    Feed-forward network with expansion and contraction layers.

    Standard transformer feed-forward network that expands input dimension by 4x,
    applies non-linearity, then contracts back to original dimension. Used in
    transformer blocks after attention layers.

    Architecture:
    - Linear expansion: n_embd → 4*n_embd
    - ReLU activation for non-linearity
    - Linear contraction: 4*n_embd → n_embd
    - Dropout for regularization

    Args:
        n_embd: Input/output embedding dimension
        dropout: Dropout probability applied after final linear layer
    """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism with optional RoPE positional embeddings.

    Implements scaled dot-product attention with multiple attention heads.
    Supports both Flash Attention (PyTorch 2.0+) and traditional attention implementations.
    Can use Rotary Position Embeddings (RoPE) for better sequence understanding,
    particularly effective for longer sequences.

    Key features:
    - Multi-head attention for capturing different attention patterns
    - Causal masking for autoregressive generation
    - Optional RoPE for position-aware attention
    - Automatic fallback between Flash Attention and traditional implementation

    Args:
        n_embd: Embedding dimension (must be divisible by n_head)
        n_head: Number of attention heads
        block_size: Maximum sequence length for masking
        dropout: Dropout probability applied to attention weights
        use_rope: Enable RoPE positional embeddings (primarily for DNA sequences)
    """
    def __init__(self, n_embd, n_head, block_size, dropout, use_rope=False):
        super().__init__()
        head_size = n_embd // n_head
        self.n_head = n_head
        self.head_size = head_size
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)
        self.flash_available = hasattr(F, 'scaled_dot_product_attention')

        # Only initialize RoPE if needed (for DNA)
        self.use_rope = use_rope
        if use_rope:
            self.rotary = RotaryEmbedding(head_size)

        # Flash attention availability logged at model level, not per-layer

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        # Apply rotary embeddings to queries and keys
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # Split heads and prepare q, k, v - ensure device consistency
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Apply RoPE only for DNA sequences
        if self.use_rope:
            cos, sin = self.rotary(x, seq_len=T)  # Pass x for device info
            q, k = self.apply_rotary_pos_emb(q, k, cos.to(x.device), sin.to(x.device))

        if self.flash_available:
            causal_mask = self.tril[:T, :T].bool()
            if mask is not None:
                if self.use_rope:
                    combined_mask = mask
                else:
                    combined_mask = torch.logical_and(
                        causal_mask.unsqueeze(0),
                        mask
                    )
            else:
                combined_mask = causal_mask.unsqueeze(0)

            attention_mask = combined_mask.float()
            attention_mask = attention_mask.masked_fill(~combined_mask, float('-inf'))
            attention_mask = attention_mask.unsqueeze(1)

            # Use flash attention with the correctly shaped mask
            if self.flash_available:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False
                )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            if mask is not None:
                att = att.masked_fill(mask[:, :T, :T].unsqueeze(1) == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for handling very long sequences
    - More effective than absolute positional embeddings for long sequences
    - Allows model to extrapolate to longer sequences than seen during training
    """
    def __init__(self, dim, max_seq_len=32768, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            # Ensure inv_freq is on the same device as x
            inv_freq = self.inv_freq.to(x.device)
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            # Scale positions for longer sequences
            if seq_len > self.max_seq_len:
                scale = math.log(seq_len / self.max_seq_len) + 1
                emb = emb / scale

            # Store cached values on the same device as input
            self.cos_cached = emb.cos()[None, None, :, :].to(x.device)
            self.sin_cached = emb.sin()[None, None, :, :].to(x.device)
        else:
            # Ensure cached values are on the correct device
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)

        return self.cos_cached, self.sin_cached


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (GQA - Grouped Query Attention) with shared Key-Value heads.

    Implements true Grouped Query Attention where multiple query heads share the same
    key and value heads, providing optimal efficiency for chess move prediction.
    Uses 4:1 ratio (n_head=8, n_kv_heads=2) for 1800 ELO chess performance.

    GQA Benefits for Chess Training (as specified in README):
    - 2-3x faster attention computation per batch vs standard MHA
    - Better convergence per 1-3 epoch training session
    - Critical for short training sessions with limited time
    - Maintains attention quality for complex chess pattern recognition
    - Reduced memory footprint while preserving chess understanding

    Chess-Specific Advantages:
    - Efficient handling of game boundary masking
    - Optimal for autoregressive move prediction
    - Balances performance and memory for Blackwell GB10 GPU (128GB)
    - Enables larger batches in short training sessions

    Args:
        n_embd: Embedding dimension (must be divisible by n_head)
        n_head: Number of query heads (attention outputs)
        n_kv_heads: Number of shared key/value heads (n_head // 4 = 4:1 GQA ratio)
        dropout: Dropout probability for attention weights
    """
    def __init__(self, n_embd, n_head, n_kv_heads, dropout):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        head_dim = n_embd // n_head
        self.n_heads = n_head
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(n_embd, n_head * head_dim)
        self.kv_proj = nn.Linear(n_embd, n_kv_heads * head_dim * 2)
        self.out_proj = nn.Linear(n_embd, n_embd)

        # QK-Norm: stabilizes attention logits, prevents explosion (Gemma 3 style)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('causal_mask', torch.tril(torch.ones(1024, 1024)))
        self.flash_available = hasattr(F, 'scaled_dot_product_attention')

        # Flash attention availability logged at model level, not per-layer

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # Project queries
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Project keys and values together
        kv = self.kv_proj(x).view(B, T, self.n_kv_heads, 2, self.head_dim)
        kv = kv.transpose(1, 2)
        k, v = kv[..., 0, :], kv[..., 1, :]

        # QK-Norm: normalize Q and K before attention to prevent logit explosion
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Expand KV heads to match query heads (view, no memory copy)
        repeat = self.n_heads // self.n_kv_heads
        k = k.unsqueeze(2).expand(B, self.n_kv_heads, repeat, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
        v = v.unsqueeze(2).expand(B, self.n_kv_heads, repeat, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)

        if self.flash_available:
            # Prepare masks
            causal_mask = self.causal_mask[:T, :T].bool()
            if mask is not None:
                game_mask = mask[:, :T, :T].bool()
                combined_mask = torch.logical_and(
                    causal_mask.unsqueeze(0),
                    game_mask
                )
            else:
                combined_mask = causal_mask.unsqueeze(0)

            attention_mask = combined_mask.unsqueeze(1)

            # Use flash attention (SDPA auto-selects optimal backend)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Fallback attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
            if mask is not None:
                att = att.masked_fill(mask[:, :T, :T].unsqueeze(1) == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) activation function.

    A gated activation function that combines Swish (SiLU) gating with linear transformations.
    Provides better gradient flow and representation capacity compared to standard ReLU
    or GELU activations, commonly used in modern transformer architectures.

    Formula: SwiGLU(x) = (SiLU(W1*x) ⊙ W2*x) @ W3
    where ⊙ is element-wise multiplication

    Advantages over ReLU/GELU:
    - Better gradient flow through gating mechanism
    - Increased model capacity without parameter explosion
    - Improved performance on complex tasks like chess

    Args:
        in_features: Input feature dimension
        hidden_features: Hidden dimension for gating (default: 4*in_features)
    """
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.w1 = nn.Linear(in_features, hidden_features)  # Gate projection
        self.w2 = nn.Linear(in_features, hidden_features)  # Value projection
        self.w3 = nn.Linear(hidden_features, in_features)  # Output projection

    def forward(self, x):
        gate = F.silu(self.w1(x))  # SiLU activation for gating
        hidden = self.w2(x)        # Linear transformation for values
        return self.w3(gate * hidden)  # Gated combination and output projection


class ChessBlock(nn.Module):
    """
    Chess-optimized transformer block with MultiQueryAttention and SwiGLU.

    A complete transformer decoder block designed specifically for chess move prediction.
    Uses RMSNorm for efficient normalization, MultiQueryAttention for memory efficiency,
    and SwiGLU activation for better gradient flow. Includes residual connections
    and dropout for training stability.

    Architecture:
    - RMSNorm pre-attention normalization
    - MultiQueryAttention with game boundary masking
    - Residual connection + dropout
    - RMSNorm pre-feedforward normalization
    - SwiGLU feed-forward network
    - Residual connection + dropout

    This block is optimized for chess where attention patterns are complex but
    memory efficiency and gradient flow are critical.

    Args:
        n_embd: Embedding dimension (model width)
        n_head: Number of query attention heads
        n_kv_heads: Number of shared key/value heads
        dropout: Dropout probability for residual connections
    """
    def __init__(self, n_embd, n_head, n_kv_heads, dropout, use_checkpoint=True):
        super().__init__()
        self.rms_1 = RMSNorm(n_embd)
        self.attn = MultiQueryAttention(n_embd, n_head, n_kv_heads, dropout)
        self.rms_2 = RMSNorm(n_embd)
        self.swiglu = SwiGLU(n_embd)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, mask=None):
        if self.training and self.use_checkpoint:
            # Gradient checkpointing for memory efficiency (recomputes forward in backward)
            def attn_checkpoint(attn_layer, rms_x, mask):
                return attn_layer(rms_x, mask=mask)

            def ffwd_checkpoint(ffwd_layer, rms_x):
                return ffwd_layer(rms_x)

            x = x + self.dropout(torch.utils.checkpoint.checkpoint(
                attn_checkpoint, self.attn, self.rms_1(x), mask, use_reentrant=False
            ))
            x = x + self.dropout(torch.utils.checkpoint.checkpoint(
                ffwd_checkpoint, self.swiglu, self.rms_2(x), use_reentrant=False
            ))
        else:
            x = x + self.dropout(self.attn(self.rms_1(x), mask=mask))
            x = x + self.dropout(self.swiglu(self.rms_2(x)))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout, use_dna=False):
        super().__init__()
        # Enable RoPE for DNA sequences
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
            use_rope=use_dna  # Only use RoPE for DNA
        )
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask=mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size, n_layer, dropout, pretrained_embeddings=None, use_chess=False, use_dna=False):
        super().__init__()
        dtype = torch.get_default_dtype()
        if pretrained_embeddings is not None:
            self.token_embedding_table = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            vocab_size, n_embd = pretrained_embeddings.shape
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd, dtype=dtype)
        self.position_embedding_table = nn.Embedding(block_size, n_embd, dtype=dtype)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, block_size, dropout, use_dna=use_dna) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.use_chess = use_chess
        self.use_dna = use_dna
        if use_chess:
            self.start_game_token = move_to_idx['<STARTGAME>']

    def create_mask(self, idx):
        if self.use_chess:
            # Existing chess game mask
            mask = torch.ones_like(idx, dtype=torch.float32)
            game_boundaries = (idx == self.start_game_token).float().cumsum(dim=1)
            mask = (game_boundaries.unsqueeze(1) == game_boundaries.unsqueeze(2)).float()
            return mask
        return None

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # Use the unified mask creation
        mask = self.create_mask(idx)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class ChessModel(nn.Module):
    """
    Chess move prediction transformer model with factorized policy heads.

    A complete transformer model specifically designed for chess move prediction
    using factorized move prediction (FROM/TO/PROMO) instead of single vocab head.
    Uses chess-specific optimizations including game boundary masking, MultiQueryAttention
    for efficiency, and RMSNorm for stable training.

    Architecture:
    - Chess move token embeddings + positional embeddings
    - Stack of ChessBlock layers (MultiQueryAttention + SwiGLU)
    - RMSNorm final normalization
    - Factorized policy heads: FROM (64), TO (64), PROMO (5)
    - Optional value head for win/draw/loss prediction

    Key chess-specific features:
    - Factorized prediction: No legality computation required
    - Game boundary masking prevents attention across game boundaries
    - Move tokenization with promotion support (e.g., e7e8q)
    - MultiQueryAttention balances performance and memory efficiency
    - Optimized for 1800+ ELO chess move prediction

    Training:
    - FROM/TO losses: Cross-entropy on move components
    - PROMO loss: Masked cross-entropy (only when promotion occurs)
    - Value loss: Optional win/draw/loss prediction

    Args:
        vocab_size: Size of chess move vocabulary (~100K with promotions)
        n_embd: Embedding dimension (model width)
        n_head: Number of query attention heads
        n_kv_heads: Number of shared key/value heads
        block_size: Maximum sequence length (chess game length)
        n_layer: Number of transformer blocks
        dropout: Dropout probability for regularization
        use_chess: Enable chess-specific masking (always True for ChessModel)
        use_dna: Enable DNA-specific features (always False for ChessModel)
    """
    def __init__(self, vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, use_chess=False, use_dna=False, token_mode='4token'):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.use_chess = use_chess
        self.use_dna = use_dna
        self.token_mode = token_mode  # 'classic' or '4token'
        if use_chess:
            self.start_game_token = move_to_idx['<STARTGAME>'] if 'move_to_idx' in globals() else None

        # Standard embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Pre-register position indices to avoid creating torch.arange every forward pass
        self.register_buffer('pos_indices', torch.arange(block_size))

        # Use ChessBlock with true GQA (Grouped Query Attention) - 4:1 ratio for efficiency
        # n_head=8, n_kv_heads=2 provides 2-3x faster attention than standard MHA
        self.blocks = nn.ModuleList([
            ChessBlock(
                n_embd=n_embd,
                n_head=n_head,
                n_kv_heads=n_kv_heads,  # 4:1 GQA ratio (n_head // 4) for optimal chess training
                dropout=dropout
            ) for _ in range(n_layer)
        ])

        # Final RMSNorm instead of LayerNorm
        self.rms_final = RMSNorm(n_embd)

        if token_mode == 'classic':
            # Classic mode: single lm_head with weight tying to token embeddings
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding_table.weight  # Weight tying
        else:
            # 4-token mode: Role-specific output heads for 4-token-per-ply grammar
            self.head_color = nn.Linear(n_embd, 2)    # White / Black
            self.head_from = nn.Linear(n_embd, 64)    # FROM square
            self.head_to = nn.Linear(n_embd, 64)      # TO square
            self.head_promo = nn.Linear(n_embd, 5)    # none / q / r / b / n

            # FROM conditioning embedding for TO prediction
            self.emb_from = nn.Embedding(64, n_embd)

        # Log flash attention once
        flash_available = hasattr(F, 'scaled_dot_product_attention')
        if flash_available:
            print(f"Flash Attention enabled ({n_layer} layers, {n_head} heads, {n_kv_heads} KV heads)")

        # Initialize weights
        self.apply(self._init_weights)

        # Scale residual output projections by 1/sqrt(2*n_layer) to prevent
        # variance growth across deep residual streams (GPT-2 convention)
        residual_scale = 1.0 / math.sqrt(2 * n_layer)
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=0.02 * residual_scale)
            torch.nn.init.normal_(block.swiglu.w3.weight, mean=0.0, std=0.02 * residual_scale)

        # Mild smoothing reduces overconfidence on deterministic Stockfish lines
        self.label_smoothing = 0.05

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, enabled):
        """Enable or disable gradient checkpointing for all ChessBlock layers."""
        for block in self.blocks:
            block.use_checkpoint = enabled
        state = "enabled" if enabled else "disabled"
        print(f"Gradient checkpointing {state} for {len(self.blocks)} layers")

    def create_game_mask(self, idx):
        """Create attention mask for chess games - exact same as Brain6"""
        if not self.use_chess:
            return None
        mask = torch.ones_like(idx, dtype=torch.float32)
        game_boundaries = (idx == self.start_game_token).float().cumsum(dim=1)
        mask = (game_boundaries.unsqueeze(1) == game_boundaries.unsqueeze(2)).float()
        return mask

    def forward(self, idx, targets=None, target_roles=None):
        B, T = idx.shape

        # Get embeddings (use pre-registered position indices)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(self.pos_indices[:T])
        x = tok_emb + pos_emb

        # Compute game mask ONCE, reuse across all layers (was computed per-layer before)
        game_mask = self.create_game_mask(idx)
        for block in self.blocks:
            x = block(x, mask=game_mask)

        # Final normalization
        x = self.rms_final(x)

        # ===== CLASSIC MODE: standard next-token prediction =====
        if self.token_mode == 'classic':
            logits = self.lm_head(x)  # [B, T, vocab_size]
            if targets is not None:
                logits_flat = logits.view(B * T, -1)
                targets_flat = targets.view(B * T)
                # Ignore PAD positions (y_roles == -1 sentinel from ClassicChessMovesDataset)
                pad_id = self.vocab_size - 3  # PAD is 3rd-to-last special token
                label_smoothing = getattr(self, 'label_smoothing', 0.0)
                loss = F.cross_entropy(logits_flat, targets_flat,
                                       ignore_index=pad_id,
                                       label_smoothing=label_smoothing)
                return logits, loss
            else:
                return logits, None

        # ===== 4-TOKEN MODE: role-specific heads (unchanged) =====
        # Training mode: route through role-specific heads based on target_roles
        if targets is not None and target_roles is not None:
            h = x.view(B * T, -1)           # [B*T, n_embd]
            targets_flat = targets.view(B * T)
            roles_flat = target_roles.view(B * T)
            idx_flat = idx.view(B * T)

            # Create role masks (skip ROLE_SPECIAL positions entirely)
            color_mask = roles_flat == ROLE_COLOR
            from_mask = roles_flat == ROLE_FROM
            to_mask = roles_flat == ROLE_TO
            promo_mask = roles_flat == ROLE_PROMO

            label_smoothing = getattr(self, 'label_smoothing', 0.0)

            # Loss weights: prioritize FROM/TO (the actual chess decisions).
            # COLOR is trivially predictable (alternates W/B) — low weight avoids
            # stealing backbone capacity from FROM/TO where it matters.
            w_color = 0.5
            w_promo = 1.0

            losses = []

            # COLOR loss (downweighted — trivially predictable, don't waste capacity)
            if color_mask.any():
                color_logits = self.head_color(h[color_mask])
                color_targets = targets_flat[color_mask] - COLOR_OFFSET
                loss_color = F.cross_entropy(color_logits, color_targets,
                                             label_smoothing=label_smoothing)
                losses.append(loss_color * w_color)

            # FROM loss (unchanged - dominant gradient signal)
            if from_mask.any():
                from_logits = self.head_from(h[from_mask])
                from_targets = targets_flat[from_mask] - FROM_OFFSET
                loss_from = F.cross_entropy(from_logits, from_targets,
                                            label_smoothing=label_smoothing)
                losses.append(loss_from)

            # TO loss (conditioned on FROM via emb_from, unchanged)
            # Stream alignment: when target is TO, input token is FROM
            if to_mask.any():
                h_to = h[to_mask]
                from_tokens = idx_flat[to_mask]
                from_local = (from_tokens - FROM_OFFSET).clamp(0, 63)
                h_to = h_to + self.emb_from(from_local)
                to_logits = self.head_to(h_to)
                to_targets = targets_flat[to_mask] - TO_OFFSET
                loss_to = F.cross_entropy(to_logits, to_targets,
                                          label_smoothing=label_smoothing)
                losses.append(loss_to)

            # PROMO loss (equal weight — mostly "none" predictions)
            if promo_mask.any():
                promo_logits = self.head_promo(h[promo_mask])
                promo_targets = targets_flat[promo_mask] - PROMO_OFFSET
                loss_promo = F.cross_entropy(promo_logits, promo_targets,
                                             label_smoothing=label_smoothing)
                losses.append(loss_promo * w_promo)

            if losses:
                loss = sum(losses) / len(losses)
            else:
                loss = torch.tensor(0.0, device=idx.device)

            # Store per-head losses for diagnostic logging
            self._last_head_losses = {
                'color': loss_color.item() if color_mask.any() else -1,
                'from': loss_from.item() if from_mask.any() else -1,
                'to': loss_to.item() if to_mask.any() else -1,
                'promo': loss_promo.item() if promo_mask.any() else -1,
            }

            return None, loss
        else:
            # Inference mode: return hidden states + head logits
            # TO logits not returned here (need per-token FROM conditioning at generation time)
            return {
                'hidden': x,
                'color': self.head_color(x),
                'from': self.head_from(x),
                'promo': self.head_promo(x),
            }, None


# Chess dataset and utility functions
class ChessMovesDataset(Dataset):
    """
    Dataset for chess games using 4-token-per-ply role-based tokenization.

    Each chess ply is represented as 4 tokens: COLOR, FROM, TO, PROMO.
    Special tokens (STARTGAME, EOFG, W, D, PAD) get ROLE_SPECIAL.

    Returns (x, y, y_roles) where y_roles tracks the role of each target token
    for routing through role-specific output heads.

    Args:
        text: Raw chess game text with games separated by blank lines
        seq_length: Length of each training sequence (context window)
        move_to_idx: Dictionary mapping token names to integer indices (140 tokens)
    """
    def __init__(self, text, seq_length, move_to_idx):
        self.seq_length = seq_length
        self.move_to_idx = move_to_idx
        self.tokens = []
        self.roles = []

        # Always use sequential tokenization to maintain correct ply counter
        # (parallel chunking breaks color alternation at chunk boundaries)
        self._tokenize_text(text)

        # Validate all tokens are within vocabulary range
        vocab_size = ROLE_VOCAB_SIZE
        invalid_tokens = [t for t in self.tokens if t >= vocab_size or t < 0]
        if invalid_tokens:
            print(f"Warning: Found {len(invalid_tokens)} invalid tokens, replacing with <PAD>")
            for i in range(len(self.tokens)):
                if self.tokens[i] < 0 or self.tokens[i] >= vocab_size:
                    self.tokens[i] = PAD
                    self.roles[i] = ROLE_SPECIAL

        # Convert to tensors
        self.tokens_tensor = torch.tensor(self.tokens, dtype=torch.long)
        self.roles_tensor = torch.tensor(self.roles, dtype=torch.long)

        # Final validation
        valid_mask = (self.tokens_tensor >= 0) & (self.tokens_tensor < vocab_size)
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum().item()
            print(f"Final validation: Found {invalid_count} invalid tokens, clamping")
            self.tokens_tensor = torch.clamp(self.tokens_tensor, 0, vocab_size - 1)

        assert len(self.tokens) == len(self.roles), f"Token/role mismatch: {len(self.tokens)} vs {len(self.roles)}"

        # Verify color alternation: sample first few games
        color_checks = 0
        for i in range(len(self.tokens)):
            if self.roles[i] == ROLE_COLOR:
                color_checks += 1
                if color_checks <= 20:
                    expected = 'W' if self.tokens[i] == COLOR_OFFSET else 'B'
                    # Don't print all, just first few
        print(f"Tokenized {len(self.tokens)} tokens, {color_checks} color tokens ({color_checks*100//max(len(self.tokens),1)}%), all validated")

    def _tokenize_text(self, text):
        """Sequential tokenization: parse text into 4-token-per-ply sequences."""
        ply = 0  # Ply counter within current game (reset at STARTGAME)
        text_len = len(text)
        next_progress = text_len // 20  # Print every 5%
        print(f"Tokenizing {text_len:,} characters...")
        i = 0
        while i < text_len:
            if i >= next_progress:
                pct = i * 100 // text_len
                print(f"  Tokenizing... {pct}% ({i:,}/{text_len:,} chars, {len(self.tokens):,} tokens)")
                next_progress += text_len // 20

            if text[i:i+11] == '<STARTGAME>':
                self.tokens.append(STARTGAME)
                self.roles.append(ROLE_SPECIAL)
                ply = 0
                i += 11
            elif text[i:i+6] == '<EOFG>':
                self.tokens.append(EOFG)
                self.roles.append(ROLE_SPECIAL)
                i += 6
            elif text[i:i+3] == '<W>':
                self.tokens.append(W_RESULT)
                self.roles.append(ROLE_SPECIAL)
                i += 3
            elif text[i:i+3] == '<D>':
                self.tokens.append(D_RESULT)
                self.roles.append(ROLE_SPECIAL)
                i += 3
            elif text[i].isspace():
                i += 1
            elif i + 4 <= len(text):
                # Try 5-char promotion move first (e7e8q)
                move_str = None
                if i + 5 <= len(text) and text[i+4].isalpha() and text[i+4].lower() in 'qrbn':
                    candidate = text[i:i+5]
                    if (candidate[0].isalpha() and candidate[1].isdigit() and
                            candidate[2].isalpha() and candidate[3].isdigit()):
                        move_str = candidate
                        i += 5

                if move_str is None:
                    candidate = text[i:i+4]
                    if (candidate[0].isalpha() and candidate[1].isdigit() and
                            candidate[2].isalpha() and candidate[3].isdigit()):
                        move_str = candidate
                        i += 4
                    else:
                        i += 1
                        continue

                # Parse UCI move into 4 tokens
                is_white = (ply % 2 == 0)
                color_tok, from_tok, to_tok, promo_tok = parse_uci_move(move_str, is_white)
                self.tokens.extend([color_tok, from_tok, to_tok, promo_tok])
                self.roles.extend([ROLE_COLOR, ROLE_FROM, ROLE_TO, ROLE_PROMO])
                ply += 1
            else:
                i += 1

    def __len__(self):
        if not hasattr(self, '_game_starts'):
            self._game_starts = torch.nonzero(
                self.tokens_tensor == STARTGAME, as_tuple=False
            ).flatten().tolist()
        return len(self._game_starts)

    def __getitem__(self, idx):
        start_pos = self._game_starts[idx]
        len_file = len(self.tokens_tensor)

        x_end = min(start_pos + self.seq_length, len_file)
        x_data = self.tokens_tensor[start_pos : x_end]

        y_end = min(start_pos + self.seq_length + 1, len_file)
        y_data = self.tokens_tensor[start_pos + 1 : y_end]
        y_roles_data = self.roles_tensor[start_pos + 1 : y_end]

        # Padded tensors
        x = torch.full((self.seq_length,), PAD, dtype=torch.long)
        y = torch.full((self.seq_length,), PAD, dtype=torch.long)
        y_roles = torch.full((self.seq_length,), ROLE_SPECIAL, dtype=torch.long)

        x[:len(x_data)] = x_data
        y[:len(y_data)] = y_data
        y_roles[:len(y_roles_data)] = y_roles_data

        return x, y, y_roles


class ClassicChessMovesDataset(Dataset):
    """
    Dataset for chess games using classic 1-token-per-move tokenization (~20K vocab).

    Each chess move is a single token. Special tokens: STARTGAME, EOFG, PAD, W, D.
    Returns (x, y, y_roles) where y_roles is all -1 (sentinel meaning 'classic mode,
    ignore roles') for training loop compatibility.
    """
    def __init__(self, text, seq_length, move_to_idx):
        self.seq_length = seq_length
        self.move_to_idx = move_to_idx
        self.tokens = []

        self._tokenize_text(text)

        # Determine PAD token
        self.pad_token = move_to_idx['<PAD>']

        # Convert to tensor
        self.tokens_tensor = torch.tensor(self.tokens, dtype=torch.long)

        # Build game start indices for __len__ / __getitem__
        startgame_id = move_to_idx['<STARTGAME>']
        self._game_starts = torch.nonzero(
            self.tokens_tensor == startgame_id, as_tuple=False
        ).flatten().tolist()

        print(f"Classic tokenized {len(self.tokens)} tokens, {len(self._game_starts)} games")

    def _tokenize_text(self, text):
        """Sequential tokenization: parse text into 1-token-per-move sequences."""
        text_len = len(text)
        next_progress = text_len // 20
        print(f"Classic tokenizing {text_len:,} characters...")
        i = 0
        while i < text_len:
            if i >= next_progress:
                pct = i * 100 // text_len
                print(f"  Tokenizing... {pct}% ({i:,}/{text_len:,} chars, {len(self.tokens):,} tokens)")
                next_progress += text_len // 20

            if text[i:i+11] == '<STARTGAME>':
                self.tokens.append(self.move_to_idx['<STARTGAME>'])
                i += 11
            elif text[i:i+6] == '<EOFG>':
                self.tokens.append(self.move_to_idx['<EOFG>'])
                i += 6
            elif text[i:i+3] == '<W>':
                self.tokens.append(self.move_to_idx['<W>'])
                i += 3
            elif text[i:i+3] == '<D>':
                self.tokens.append(self.move_to_idx['<D>'])
                i += 3
            elif text[i].isspace():
                i += 1
            elif i + 4 <= text_len:
                # Try 5-char promotion move first (e7e8q -> E7E8Q)
                move_str = None
                if i + 5 <= text_len and text[i+4].isalpha() and text[i+4].lower() in 'qrbn':
                    candidate = text[i:i+5].upper()
                    if (candidate[0].isalpha() and candidate[1].isdigit() and
                            candidate[2].isalpha() and candidate[3].isdigit()):
                        if candidate in self.move_to_idx:
                            move_str = candidate
                            i += 5

                if move_str is None:
                    candidate = text[i:i+4].upper()
                    if (candidate[0].isalpha() and candidate[1].isdigit() and
                            candidate[2].isalpha() and candidate[3].isdigit()):
                        if candidate in self.move_to_idx:
                            move_str = candidate
                            i += 4
                        else:
                            i += 1
                            continue
                    else:
                        i += 1
                        continue

                self.tokens.append(self.move_to_idx[move_str])
            else:
                i += 1

    def __len__(self):
        return len(self._game_starts)

    def __getitem__(self, idx):
        start_pos = self._game_starts[idx]
        len_file = len(self.tokens_tensor)

        x_end = min(start_pos + self.seq_length, len_file)
        x_data = self.tokens_tensor[start_pos:x_end]

        y_end = min(start_pos + self.seq_length + 1, len_file)
        y_data = self.tokens_tensor[start_pos + 1:y_end]

        # Padded tensors
        x = torch.full((self.seq_length,), self.pad_token, dtype=torch.long)
        y = torch.full((self.seq_length,), self.pad_token, dtype=torch.long)
        # y_roles = -1 sentinel means "classic mode, no role routing"
        y_roles = torch.full((self.seq_length,), -1, dtype=torch.long)

        x[:len(x_data)] = x_data
        y[:len(y_data)] = y_data

        return x, y, y_roles


class _PreTokenizedDataset(Dataset):
    """Wraps pre-tokenized tensors to avoid re-tokenizing in each DDP worker process."""
    def __init__(self, tokens_tensor, roles_tensor, seq_length, token_mode, move_to_idx):
        self.tokens_tensor = tokens_tensor
        self.roles_tensor = roles_tensor
        self.seq_length = seq_length
        self.token_mode = token_mode

        if token_mode == 'classic':
            self.pad_token = move_to_idx['<PAD>']
            startgame_id = move_to_idx['<STARTGAME>']
        else:
            self.pad_token = PAD
            startgame_id = STARTGAME

        self._game_starts = torch.nonzero(
            self.tokens_tensor == startgame_id, as_tuple=False
        ).flatten().tolist()

    def __len__(self):
        return len(self._game_starts)

    def __getitem__(self, idx):
        start_pos = self._game_starts[idx]
        len_file = len(self.tokens_tensor)

        x_end = min(start_pos + self.seq_length, len_file)
        x_data = self.tokens_tensor[start_pos:x_end]

        y_end = min(start_pos + self.seq_length + 1, len_file)
        y_data = self.tokens_tensor[start_pos + 1:y_end]

        x = torch.full((self.seq_length,), self.pad_token, dtype=torch.long)
        y = torch.full((self.seq_length,), self.pad_token, dtype=torch.long)

        x[:len(x_data)] = x_data
        y[:len(y_data)] = y_data

        if self.token_mode == 'classic':
            y_roles = torch.full((self.seq_length,), -1, dtype=torch.long)
        else:
            y_roles_data = self.roles_tensor[start_pos + 1:y_end]
            y_roles = torch.full((self.seq_length,), ROLE_SPECIAL, dtype=torch.long)
            y_roles[:len(y_roles_data)] = y_roles_data

        return x, y, y_roles


def process_chunk_for_chess_moves(args):
    """Parallel tokenization worker for 4-token-per-ply grammar."""
    (chunk_text,) = args
    chunk_tokens = []
    chunk_roles = []
    ply = 0
    i = 0
    while i < len(chunk_text):
        if chunk_text[i:i+11] == '<STARTGAME>':
            chunk_tokens.append(STARTGAME)
            chunk_roles.append(ROLE_SPECIAL)
            ply = 0
            i += 11
        elif chunk_text[i:i+6] == '<EOFG>':
            chunk_tokens.append(EOFG)
            chunk_roles.append(ROLE_SPECIAL)
            i += 6
        elif chunk_text[i:i+3] == '<W>':
            chunk_tokens.append(W_RESULT)
            chunk_roles.append(ROLE_SPECIAL)
            i += 3
        elif chunk_text[i:i+3] == '<D>':
            chunk_tokens.append(D_RESULT)
            chunk_roles.append(ROLE_SPECIAL)
            i += 3
        elif chunk_text[i].isspace():
            i += 1
        elif i + 4 <= len(chunk_text):
            move_str = None
            if i + 5 <= len(chunk_text) and chunk_text[i+4].isalpha() and chunk_text[i+4].lower() in 'qrbn':
                candidate = chunk_text[i:i+5]
                if (candidate[0].isalpha() and candidate[1].isdigit() and
                        candidate[2].isalpha() and candidate[3].isdigit()):
                    move_str = candidate
                    i += 5

            if move_str is None:
                candidate = chunk_text[i:i+4]
                if (candidate[0].isalpha() and candidate[1].isdigit() and
                        candidate[2].isalpha() and candidate[3].isdigit()):
                    move_str = candidate
                    i += 4
                else:
                    i += 1
                    continue

            is_white = (ply % 2 == 0)
            color_tok, from_tok, to_tok, promo_tok = parse_uci_move(move_str, is_white)
            chunk_tokens.extend([color_tok, from_tok, to_tok, promo_tok])
            chunk_roles.extend([ROLE_COLOR, ROLE_FROM, ROLE_TO, ROLE_PROMO])
            ply += 1
        else:
            i += 1
    return chunk_tokens, chunk_roles


def uci_to_square(file_char, rank_char):
    """Convert UCI file/rank chars to square index 0..63 (a8=0, h1=63)."""
    file_idx = ord(file_char.lower()) - ord('a')  # 0-7
    rank_idx = int(rank_char)                      # 1-8
    row = 8 - rank_idx                             # rank 8 -> row 0, rank 1 -> row 7
    return row * 8 + file_idx


def square_to_uci(sq):
    """Convert square index 0..63 to UCI string like 'e2'."""
    file_char = chr(ord('a') + (sq % 8))
    rank_char = str(8 - (sq // 8))
    return f"{file_char}{rank_char}"


def parse_uci_move(move_str, is_white):
    """Parse a UCI move string into 4 global token IDs (COLOR, FROM, TO, PROMO)."""
    move_str = move_str.lower().strip()
    from_sq = uci_to_square(move_str[0], move_str[1])
    to_sq = uci_to_square(move_str[2], move_str[3])

    promo_map = {'q': 1, 'r': 2, 'b': 3, 'n': 4}
    if len(move_str) >= 5 and move_str[4] in promo_map:
        promo_idx = promo_map[move_str[4]]
    else:
        promo_idx = 0  # No promotion

    color_tok = COLOR_OFFSET + (0 if is_white else 1)
    from_tok = FROM_OFFSET + from_sq
    to_tok = TO_OFFSET + to_sq
    promo_tok = PROMO_OFFSET + promo_idx

    return color_tok, from_tok, to_tok, promo_tok


def global_to_role_local(global_id):
    """Convert a global token ID to (role, local_class_index)."""
    if global_id < FROM_OFFSET:
        return ROLE_COLOR, global_id - COLOR_OFFSET
    elif global_id < TO_OFFSET:
        return ROLE_FROM, global_id - FROM_OFFSET
    elif global_id < PROMO_OFFSET:
        return ROLE_TO, global_id - TO_OFFSET
    elif global_id < STARTGAME:
        return ROLE_PROMO, global_id - PROMO_OFFSET
    else:
        return ROLE_SPECIAL, global_id


def role_local_to_global(role, local_id):
    """Convert a role and local class index back to a global token ID."""
    if role == ROLE_COLOR:
        return COLOR_OFFSET + local_id
    elif role == ROLE_FROM:
        return FROM_OFFSET + local_id
    elif role == ROLE_TO:
        return TO_OFFSET + local_id
    elif role == ROLE_PROMO:
        return PROMO_OFFSET + local_id
    else:
        return local_id


def create_move_to_idx():
    """Create the 140-token role-based vocabulary mapping."""
    move_to_idx = {}

    # Color tokens (0-1)
    move_to_idx['<WHITE>'] = COLOR_OFFSET + 0
    move_to_idx['<BLACK>'] = COLOR_OFFSET + 1

    # FROM square tokens (2-65)
    for sq in range(64):
        name = f'F:{square_to_uci(sq)}'
        move_to_idx[name] = FROM_OFFSET + sq

    # TO square tokens (66-129)
    for sq in range(64):
        name = f'T:{square_to_uci(sq)}'
        move_to_idx[name] = TO_OFFSET + sq

    # PROMO tokens (130-134)
    for i, label in enumerate(['none', 'q', 'r', 'b', 'n']):
        move_to_idx[f'<PROMO:{label}>'] = PROMO_OFFSET + i

    # Special tokens (135-139)
    move_to_idx['<STARTGAME>'] = STARTGAME
    move_to_idx['<EOFG>'] = EOFG
    move_to_idx['<PAD>'] = PAD
    move_to_idx['<W>'] = W_RESULT
    move_to_idx['<D>'] = D_RESULT

    return move_to_idx


def create_idx_to_move():
    idx_to_move = {idx: move for move, idx in move_to_idx.items()}
    return idx_to_move


def create_classic_move_to_idx():
    """Create classic ~20K vocab: 64*63*5 move tokens + 5 special.

    Each move is encoded as a single token:
      move_id = from_sq * 315 + to_offset * 5 + promo_idx
    where to_offset compresses TO into 0..62 by skipping FROM.
    Total: 20,160 move tokens + 5 special = 20,165.
    """
    m = {}
    for from_sq in range(64):
        from_file = chr(97 + (from_sq % 8))
        from_rank = str(8 - (from_sq // 8))
        for to_sq in range(64):
            if to_sq == from_sq:
                continue
            to_file = chr(97 + (to_sq % 8))
            to_rank = str(8 - (to_sq // 8))
            to_offset = to_sq if to_sq < from_sq else (to_sq - 1)
            for promo_idx, promo_char in enumerate(['', 'q', 'r', 'b', 'n']):
                move_id = (from_sq * 63 * 5) + (to_offset * 5) + promo_idx
                move_str = f"{from_file}{from_rank}{to_file}{to_rank}{promo_char}".upper()
                m[move_str] = move_id
    # Special tokens start after move tokens
    for idx, token in enumerate(['<STARTGAME>', '<EOFG>', '<PAD>', '<W>', '<D>'], start=len(m)):
        m[token] = idx
    return m


def create_classic_idx_to_move(classic_move_to_idx):
    """Reverse mapping for classic tokenizer."""
    return {idx: move for move, idx in classic_move_to_idx.items()}


_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)


def _result_to_token(result_str):
    """Convert game result string to token marker."""
    if result_str == "1-0":
        return "<W>"
    # Treat 0-1, 1/2-1/2, draws, and anything else as <D>
    return "<D>"


def _read_parquet_as_text(file_path):
    """Read parquet file and produce same format as .txt: '<W> e2e4 e7e5...\\n\\n<D> d2d4...'"""
    print(f"Loading parquet file: {file_path}")
    df = pd.read_parquet(file_path, columns=['Moves', 'Result'])
    total = len(df)
    print(f"Found {total:,} rows, converting to text...")
    games = []
    skipped = 0
    last_pct = -1
    for i, (_, row) in enumerate(df.iterrows()):
        # Progress update every 1%
        pct = (i * 100) // total
        if pct != last_pct:
            last_pct = pct
            print(f"\rConverting games: {pct}% ({i:,}/{total:,})", end='', flush=True)
        moves = row.get('Moves', None)
        if moves is None:
            skipped += 1
            continue
        # Handle numpy arrays, lists, tuples, and other iterables
        try:
            move_list = list(moves)
        except (TypeError, ValueError):
            skipped += 1
            continue
        cleaned = [str(m).strip().upper() for m in move_list if _UCI_RE.match(str(m).strip())]
        if cleaned:
            games.append(f"{_result_to_token(str(row.get('Result', '')).strip())} {' '.join(cleaned)}")
        else:
            skipped += 1
    print(f"\rConverting games: 100% ({total:,}/{total:,})")
    print(f"Read {len(games):,} games from parquet file ({skipped:,} skipped)")
    return '\n\n'.join(games)


def load_chess_file():
    """Load chess games file for training"""
    print("Please select a chess games file.")
    file_path = filedialog.askopenfilename(
        title="Select Chess Games File",
        filetypes=[("Chess files", "*.txt *.parquet"), ("Text files", "*.txt"), ("Parquet files", "*.parquet")]
    )

    if file_path:
        try:
            if file_path.lower().endswith('.parquet'):
                text = _read_parquet_as_text(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            print(f"Chess games file loaded: {file_path}")
            print(f"Total characters: {len(text)}")

            # Process games with win/draw markers
            games = text.split('\n\n')
            processed_games = []
            for game in games:
                game = game.strip()
                if not game:
                    continue
                # Extract result marker (<W> or <D>) and moves
                if game.startswith('<W> ') or game.startswith('<D> '):
                    marker = game[:3]  # <W> or <D>
                    moves = game[4:]  # Rest is moves
                    processed_games.append(f'<STARTGAME> {marker} {moves} <EOFG>')
                else:
                    # Legacy format: assume all games are draws for now
                    processed_games.append(f'<STARTGAME> <D> {game} <EOFG>')
            text = '\n'.join(processed_games)
            print(f"Processed {len(processed_games)} chess games with result markers")

            return text, file_path
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None
    else:
        print("No file selected.")
        return None, None


def get_input_with_default(prompt, default_value):
    try:
        value = input(f"{prompt} (default: {default_value}): ")
        return value if value.strip() else default_value
    except EOFError:
        # Handle non-interactive environments
        print(f"{prompt} (default: {default_value}): {default_value}")
        return default_value


def create_file_dialog(title="Select File", filetypes=None, initialdir=None):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes,
        initialdir=initialdir
    )
    root.destroy()
    return file_path


def save_token_embeddings(model, filepath):
    model_module = model.module if isinstance(model, nn.DataParallel) else model
    embeddings = model_module.token_embedding_table.weight.data
    torch.save(embeddings, filepath)
    filename = os.path.basename(filepath)
    model_folder = os.path.dirname(filepath)
    print(f"Token embedding saved: {filename} in {model_folder}")


def save_model_all(model, all_text, n_embd, n_head, n_kv_heads, n_layer, dropout, block_size, epoch, batch_idx, batch_size, optimizer, scheduler, scaler, loss, learning_rate=None, weight_decay=None, gpu_indices=None, token_mode='4token'):
    """
    Save complete chess model checkpoint with all training state.

    Creates a comprehensive checkpoint containing model weights, training state,
    and generated samples. Optimized for chess model resumption and analysis.

    Saves:
    - Model architecture and trained weights
    - Optimizer state(s) (Adafactor) for training continuation
    - Learning rate scheduler state(s)
    - Gradient scaler state(s) for mixed precision
    - Training progress (epoch, batch, hyperparameters)
    - GPU configuration used for training
    - Sample generated chess moves for progress tracking
    - Token embeddings separately for analysis

    File naming: C{n_layer}H{n_head}E{n_embd}_B{batch_size}_E{epoch}B{batch}_L{loss}_{timestamp}.pth

    Args:
        optimizer: Single optimizer dict OR list of optimizer dicts (for multi-GPU)
        scheduler: Single scheduler dict OR list of scheduler dicts (for multi-GPU)
        scaler: Single scaler dict OR list of scaler dicts (for multi-GPU)
        (other args same as before)
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    # Try original data path first, fallback to /data/Data if not available
    if platform.system() == "Darwin":
        BASE_DIR = "/Users/jonathanrothberg/Data"
    else:
        original_path = "/home/jonathan/Data"
        fallback_path = "/data/Data"
        BASE_DIR = original_path if os.path.exists(original_path) else fallback_path

    # Ensure the base directory exists
    os.makedirs(BASE_DIR, exist_ok=True)

    model_prefix = "C"  # Chess LLM
    model_id = f"{model_prefix}{n_layer}H{n_head}E{n_embd}"
    if n_kv_heads != n_head:
        model_id += f"K{n_kv_heads}"

    folder_prefix = "Chess"
    model_folder = os.path.join(BASE_DIR, f"{folder_prefix}_Model_{model_id}")
    model_filename = f"{model_id}_B{batch_size}_E{epoch+1}B{batch_idx+1}_L{loss:.3f}_{timestamp}.pth"

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_module = model.module if isinstance(model, nn.DataParallel) else model
    model_state_dict = model_module.state_dict()

    # Handle both single objects and lists (for multi-GPU)
    if isinstance(optimizer, list):
        # Check if list contains optimizer objects or state dicts
        if len(optimizer) > 0 and hasattr(optimizer[0], 'state_dict'):
            # List of optimizer objects - call state_dict()
            optimizer_state = [opt.state_dict() for opt in optimizer]
        else:
            # List of state dicts (from checkpoint loading) - use directly
            optimizer_state = optimizer

        if len(scheduler) > 0 and hasattr(scheduler[0], 'state_dict'):
            scheduler_state = [sched.state_dict() for sched in scheduler]
        else:
            scheduler_state = scheduler

        if scaler and len(scaler) > 0:
            if hasattr(scaler[0], 'state_dict'):
                scaler_state = [scal.state_dict() for scal in scaler]
            else:
                scaler_state = scaler
        else:
            scaler_state = None
    else:
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()
        scaler_state = scaler.state_dict() if scaler else None

    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state,
        'scheduler_state_dict': scheduler_state,
        'scaler_state_dict': scaler_state,
        'epoch': epoch,
        'batch_idx': batch_idx,
        'hyperparameters': {
            'vocab_size': len(move_to_idx),
            'format_version': 3 if token_mode == 'classic' else 2,
            'token_mode': token_mode,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_kv_heads': n_kv_heads,
            'n_layer': n_layer,
            'dropout': dropout,
            'block_size': block_size,
            'label_smoothing': getattr(model_module, 'label_smoothing', 0.0),
            'use_chess': True,
            'use_dna': False,
            # Training parameters (can be changed when reloading)
            'batch_size': batch_size,
            'learning_rate': learning_rate if learning_rate is not None else 3e-4,
            'weight_decay': weight_decay if weight_decay is not None else 0.01,
            'gpu_indices': gpu_indices,
        },
        'tokenizer': move_to_idx,
        'dataset_type': 'chess_moves'
    }

    print(f"\nModel folder: {model_folder}, {checkpoint['hyperparameters']}")
    torch.save(checkpoint, os.path.join(model_folder, model_filename))
    print(f"Model saved: {model_filename} in {model_folder}")

    # COMMENTED OUT: Save generated text samples (uncomment if needed for analysis)
    # filenamealltext = f"all_text_{epoch+1}_{timestamp}.txt"
    # with open(os.path.join(model_folder, filenamealltext), 'w', encoding='utf-8') as file:
    #     file.write(all_text)
    # print(f"Text saved: {filenamealltext} in {model_folder}")

    # COMMENTED OUT: Save token embeddings separately (uncomment if needed for analysis)
    # embedding_filename = f"token_embeddings_{timestamp}.pt"
    # save_token_embeddings(model, os.path.join(model_folder, embedding_filename))


def test_progress(epoch, num_epochs, batch_idx, data_loader, loss, model, x, tokens_to_generate, all_text, idx_to_move):
    """
    Generate sample chess moves during training to monitor model progress.

    Supports both classic (1-token) and 4-token-per-ply generation modes.
    """
    print(f"\nEpoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss:.4f}")

    was_training = model.training
    model.eval()
    model_single = model.module if isinstance(model, nn.DataParallel) else model

    # Detect token mode
    model_raw = model_single._orig_mod if hasattr(model_single, '_orig_mod') else model_single
    token_mode = getattr(model_raw, 'token_mode', '4token')

    # Print per-head losses for 4token mode diagnostics
    if token_mode == '4token':
        model_raw_diag = model_raw
        if hasattr(model_raw_diag, '_last_head_losses'):
            hl = model_raw_diag._last_head_losses
            print(f"  Head losses -> color: {hl['color']:.4f}  from: {hl['from']:.4f}  to: {hl['to']:.4f}  promo: {hl['promo']:.4f}")

    with torch.no_grad():
        input_seq = x[-1].unsqueeze(0)
        # Display input sequence using idx_to_move
        input_seq_str = ' '.join([idx_to_move.get(t.item(), f'<UNK:{t.item()}>') for t in input_seq[0]])

        print("\nInput Sequence:")
        print(input_seq_str)

        dev = input_seq.device

        if token_mode == 'classic':
            # === CLASSIC MODE: simple autoregressive generation ===
            generated_moves = []
            num_moves = min(tokens_to_generate, 32)
            # Find special token IDs to skip during generation
            special_ids = set()
            for name in ['<STARTGAME>', '<EOFG>', '<PAD>', '<W>', '<D>']:
                if name in move_to_idx:
                    special_ids.add(move_to_idx[name])

            for _ in range(num_moves):
                logits, _ = model_single(input_seq)
                next_logits = logits[0, -1]  # [vocab_size]
                # Mask out special tokens
                for sid in special_ids:
                    next_logits[sid] = float('-inf')
                next_tok = next_logits.argmax(dim=-1).item()
                move_name = idx_to_move.get(next_tok, f'<UNK:{next_tok}>')
                generated_moves.append(move_name)
                input_seq = torch.cat([input_seq[:, 1:],
                                       torch.tensor([[next_tok]], device=dev)], dim=1)

            generated_text = ' '.join(generated_moves)

        else:
            # === 4-TOKEN MODE: 4-step role-specific generation (unchanged) ===
            generated_moves = []
            num_plies = min(tokens_to_generate, 32)

            # Trim input to last complete ply boundary
            tokens_list = input_seq[0].tolist()
            trim_to = len(tokens_list)
            for j in range(len(tokens_list) - 1, -1, -1):
                t = tokens_list[j]
                if (PROMO_OFFSET <= t < PROMO_OFFSET + 5) or t in (STARTGAME, EOFG, W_RESULT, D_RESULT, PAD):
                    trim_to = j + 1
                    break
            if trim_to < len(tokens_list):
                input_seq = input_seq[:, :trim_to]
                pad_len = x.shape[-1] - trim_to
                if pad_len > 0:
                    padding = torch.full((1, pad_len), PAD, dtype=torch.long, device=dev)
                    input_seq = torch.cat([padding, input_seq], dim=1)

            for _ in range(num_plies):
                # 1. COLOR prediction
                output, _ = model_single(input_seq)
                color_logits = output['color'][0, -1]
                color_idx = color_logits.argmax(dim=-1).item()
                color_tok = COLOR_OFFSET + color_idx
                input_seq = torch.cat([input_seq[:, 1:],
                                       torch.tensor([[color_tok]], device=dev)], dim=1)

                # 2. FROM prediction
                output, _ = model_single(input_seq)
                from_logits = output['from'][0, -1]
                from_sq = from_logits.argmax(dim=-1).item()
                from_tok = FROM_OFFSET + from_sq
                input_seq = torch.cat([input_seq[:, 1:],
                                       torch.tensor([[from_tok]], device=dev)], dim=1)

                # 3. TO prediction (conditioned on FROM)
                output, _ = model_single(input_seq)
                h_last = output['hidden'][0, -1]
                from_emb = model_raw.emb_from(torch.tensor(from_sq, device=dev))
                h_conditioned = h_last + from_emb
                to_logits = model_raw.head_to(h_conditioned)
                to_logits[from_sq] = float('-inf')
                to_sq = to_logits.argmax(dim=-1).item()
                to_tok = TO_OFFSET + to_sq
                input_seq = torch.cat([input_seq[:, 1:],
                                       torch.tensor([[to_tok]], device=dev)], dim=1)

                # 4. PROMO prediction
                output, _ = model_single(input_seq)
                promo_logits = output['promo'][0, -1]
                promo_idx = promo_logits.argmax(dim=-1).item()
                promo_tok = PROMO_OFFSET + promo_idx
                input_seq = torch.cat([input_seq[:, 1:],
                                       torch.tensor([[promo_tok]], device=dev)], dim=1)

                # Reconstruct UCI move string
                from_str = square_to_uci(from_sq)
                to_str = square_to_uci(to_sq)
                promo_chars = ['', 'q', 'r', 'b', 'n']
                promo_str = promo_chars[promo_idx] if promo_idx < len(promo_chars) else ''
                color_str = 'W' if color_idx == 0 else 'B'
                move_str = f"{color_str}:{from_str}{to_str}{promo_str}"
                generated_moves.append(move_str)

            generated_text = ' '.join(generated_moves)

        print("\nGenerated Moves:")
        print(generated_text)

        all_text = all_text + ("\nInput Sequence:\n" + input_seq_str +
                               "\nGenerated Moves:\n" + generated_text)

    if was_training:
        model.train()

    return all_text


def load_model_file(model_file_path=None):
    """
    Load saved chess model checkpoint for inference or training resumption.

    Loads a complete chess model checkpoint including architecture, weights,
    training state, and chess tokenization. Handles DataParallel compatibility
    and device placement automatically.

    Loading process:
    1. File selection via GUI or direct path
    2. Checkpoint loading with device compatibility
    3. Model architecture reconstruction from saved hyperparameters
    4. Weight loading with DataParallel prefix handling
    5. Chess tokenizer restoration
    6. Device placement and optimization

    Returns:
        Tuple for training resumption: (model, vocab_size, n_embd, n_head, n_kv_heads,
                                       block_size, n_layer, dropout, optimizer_state_dict,
                                       scheduler_state_dict, scaler_state_dict, last_epoch,
                                       last_batch_idx, hyperparameters)

    Args:
        model_file_path: Optional direct path to model file (if None, shows file dialog)
    """
    if model_file_path is None:
        print("Please select a chess model file.")
        # Prefer /data/Data for checkpoint picker when present; else /home/jonathan/Data.
        checkpoint_dialog_dir = "/data/Data" if os.path.isdir("/data/Data") else "/home/jonathan/Data"
        print(f"Opening file dialog in: {checkpoint_dialog_dir}")
        model_file = create_file_dialog(title="Select Chess Model File", filetypes=[("PyTorch files", "*.pth")], initialdir=checkpoint_dialog_dir)
    else:
        model_file = model_file_path
        print(f"Loading model file: {model_file}")

    if model_file:
        # Load checkpoint
        if device.type == 'mps':
            checkpoint = torch.load(model_file, map_location='cpu')
        else:
            checkpoint = torch.load(model_file)

        # Get hyperparameters
        hyperparameters = checkpoint['hyperparameters']

        # Detect token mode from checkpoint
        token_mode = hyperparameters.get('token_mode', '4token')
        fmt_version = hyperparameters.get('format_version', 1)

        # Reject truly ancient checkpoints (format_version 1 without token_mode)
        if fmt_version < 2 and token_mode == '4token':
            print("ERROR: Old checkpoint format (composite move tokens) not compatible with role-specific heads.")
            print("       This checkpoint uses the ~20K composite vocabulary. Re-train with the new 4-token-per-ply format.")
            return None

        vocab_size = hyperparameters['vocab_size']
        n_embd = hyperparameters['n_embd']
        n_head = hyperparameters['n_head']
        n_layer = hyperparameters['n_layer']
        dropout = hyperparameters['dropout']
        block_size = hyperparameters['block_size']

        # Ensure n_embd is divisible by n_head for attention layers
        if n_embd % n_head != 0:
            print(f"Warning: n_embd ({n_embd}) not divisible by n_head ({n_head}), adjusting n_embd")
            original_embd = n_embd
            while n_embd % n_head != 0:
                n_embd += n_head
            print(f"Adjusted n_embd from {original_embd} to {n_embd} (head_dim = {n_embd // n_head})")

        n_kv_heads = hyperparameters.get('n_kv_heads', n_head // 4)

        # Load tokenizer — set globals based on mode
        global move_to_idx, idx_to_move
        tokenizer = checkpoint.get('tokenizer')
        if isinstance(tokenizer, dict):
            move_to_idx = tokenizer
            idx_to_move = {idx: move for move, idx in move_to_idx.items()}
            print(f"Loaded {token_mode} tokenizer with {len(move_to_idx)} tokens")
        elif token_mode == 'classic':
            move_to_idx = create_classic_move_to_idx()
            idx_to_move = create_classic_idx_to_move(move_to_idx)
            vocab_size = len(move_to_idx)
            print(f"Created fresh classic tokenizer with {len(move_to_idx)} tokens")
        else:
            move_to_idx = create_move_to_idx()
            idx_to_move = create_idx_to_move()

        # Create chess model with correct mode
        print(f"Creating ChessModel in '{token_mode}' mode (vocab_size={vocab_size})")
        model = ChessModel(vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout,
                           use_chess=True, token_mode=token_mode)
        model.start_game_token = move_to_idx['<STARTGAME>']

        # Move model to device before loading state dict
        model = model.to(device)

        # Load state dict with proper handling
        state_dict = checkpoint['model_state_dict']
        cleaned_state_dict = {}
        for key, val in state_dict.items():
            new_key = key
            if new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
            elif new_key.startswith('_orig_mod.module.'):
                new_key = new_key[len('_orig_mod.module.'):]
            elif new_key.startswith('_orig_mod.'):
                new_key = new_key[len('_orig_mod.'):]
            cleaned_state_dict[new_key] = val

        model.load_state_dict(cleaned_state_dict, strict=False)

        # NOTE: torch.compile is applied LATER in _train_chess_model_core after GPU setup
        # Do NOT apply torch.compile here - it breaks .parameters() and .state_dict() access

        # Get optimizer and scheduler states
        optimizer_state_dict = checkpoint.get('optimizer_state_dict')
        scheduler_state_dict = checkpoint.get('scheduler_state_dict')
        scaler_state_dict = checkpoint.get('scaler_state_dict')
        last_epoch = checkpoint.get('epoch', -1)
        last_batch_idx = checkpoint.get('batch_idx', -1)

        model_filename = os.path.basename(model_file)
        print(f"Chess model loaded: {model_filename} from {os.path.dirname(model_file)}")
        print(f"Model hyperparameters: {hyperparameters}")
        print(f"Last epoch: {last_epoch}, Last batch: {last_batch_idx}")

        return (model, vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout,
                optimizer_state_dict, scheduler_state_dict, scaler_state_dict, last_epoch, last_batch_idx, hyperparameters)
    else:
        print("No model file selected.")
        return None


def get_model_module(model):
    """Helper function to safely access the model when it might be wrapped in DataParallel"""
    if isinstance(model, nn.DataParallel) or hasattr(model, 'module'):
        return model.module
    return model


def select_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        print(f"Only {num_gpus} GPU available. Using it for computation.")
        return list(range(num_gpus))

    all_gpus = list(range(num_gpus))
    default_gpus = ",".join(str(i) for i in all_gpus)

    print(f"Available GPUs: {num_gpus}")
    print("Note: Single GPU (0) is most reliable")
    custom_gpus = input(f"Enter GPU indices separated by commas (default: {default_gpus}): ")

    if not custom_gpus.strip():
        print(f"Using all {num_gpus} GPUs")
        return all_gpus

    try:
        gpu_indices = [int(idx.strip()) for idx in custom_gpus.split(',')]
        if all(0 <= idx < num_gpus for idx in gpu_indices):
            if len(gpu_indices) == 1 and gpu_indices[0] == 0:
                print("Selected GPU 0 only - will skip DataParallel for reliable training")
            return gpu_indices
        else:
            print(f"Invalid GPU index. Using all available GPUs.")
            return all_gpus
    except ValueError:
        print(f"Invalid input. Using all available GPUs.")
        return all_gpus


def enter_batch_size(n_embd, n_head, block_size, n_layer, batch_size, gpu_indices, n_kv_heads=None):
    """Calculate batch size based on GPU memory and model architecture."""
    bytes_per_float = 4
    num_gpus = len(gpu_indices)

    head_dim = n_embd // n_head
    if n_kv_heads is None:
        n_kv_heads = max(1, n_head // 4)

    # --- Model static memory (per GPU replica) ---
    vocab_size = len(move_to_idx)
    token_emb = vocab_size * n_embd
    pos_emb = block_size * n_embd
    attn_per_layer = n_embd * n_embd + n_embd * n_kv_heads * head_dim * 2 + n_embd * n_embd  # Q+KV+Out
    ffn_per_layer = 12 * n_embd * n_embd  # SwiGLU: 3 matrices x (n_embd x 4*n_embd)
    rms_per_layer = 2 * n_embd
    total_params = token_emb + pos_emb + n_layer * (attn_per_layer + ffn_per_layer + rms_per_layer)

    model_bytes = total_params * bytes_per_float
    optimizer_bytes = model_bytes * 2  # Adam: momentum + variance
    gradient_bytes = model_bytes
    static_per_gpu = model_bytes + optimizer_bytes + gradient_bytes

    # --- Per-sequence activation memory ---
    # Stored per layer: checkpoint input (FP16) + backward gradient (FP32)
    stored_per_layer = block_size * n_embd * (2 + bytes_per_float)
    stored_total = n_layer * stored_per_layer

    # Peak recompute (one layer at a time): attention + SwiGLU FFN intermediates (FP16)
    attn_peak = block_size * n_embd * 4 * 2       # Q, K, V, output
    ffn_peak = block_size * (4 * n_embd) * 3 * 2  # gate, up, product
    recompute_peak = attn_peak + ffn_peak

    # Embeddings
    embedding_mem = block_size * n_embd * 2 * 2

    raw_per_seq = embedding_mem + stored_total + recompute_peak

    # Practical overhead: autograd graph, caching allocator fragmentation,
    # torch.compile kernel cache, temporary tensors, mixed precision buffers
    total_per_seq = int(raw_per_seq * 4.0)

    # --- GPU memory: use actual free memory (accounts for desktop/display processes) ---
    if torch.cuda.is_available() and len(gpu_indices) > 0:
        print(f"\n  GPU memory (actual free):")
        gpu_free = []
        for i in gpu_indices:
            free, total = torch.cuda.mem_get_info(i)
            other_used = total - free
            gpu_free.append(free)
            if other_used > 100 * 1024**2:  # >100MB used by other processes
                print(f"    GPU {i}: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total ({other_used/1e9:.1f} GB used by other processes)")
            else:
                print(f"    GPU {i}: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
        # Bottleneck = GPU with least free memory (usually the display GPU)
        per_gpu_free = min(gpu_free)
    else:
        per_gpu_free = 64 * 1024**3

    per_gpu_available = per_gpu_free * 0.90 - static_per_gpu  # 90% of free, minus static
    max_seqs_per_gpu = max(1, int(per_gpu_available / total_per_seq))
    max_batch_size = max_seqs_per_gpu * num_gpus

    print(f"\n  Memory estimate:")
    print(f"    Model + optimizer + gradients per GPU: {static_per_gpu / 1e9:.2f} GB")
    print(f"    Activation memory per sequence: {total_per_seq / 1e6:.0f} MB")
    print(f"    Bottleneck GPU free: {per_gpu_free / 1e9:.1f} GB")
    print(f"    Available for activations: {per_gpu_available / 1e9:.1f} GB")
    print(f"    Max batch size: {max_batch_size} ({max_seqs_per_gpu} per GPU x {num_gpus} GPUs)")

    # Recommend 75% of max for safety
    recommended_batch = max(1, (max_batch_size * 3 // 4))
    # Round down to nearest multiple of num_gpus for even splits
    recommended_batch = (recommended_batch // num_gpus) * num_gpus
    recommended_batch = max(num_gpus, recommended_batch)

    batch_size = int(get_input_with_default(f"Enter batch size (recommended: {recommended_batch}, max: {max_batch_size})", recommended_batch))
    batch_size = max(1, min(batch_size, max_batch_size))

    return batch_size


# Global variables for chess tokenization
move_to_idx = create_move_to_idx()
idx_to_move = create_idx_to_move()


def _ddp_setup(rank, world_size):
    """Initialize the distributed process group for DDP training."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _ddp_cleanup():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()





# Core training function — DDP multi-GPU version
# Uses mp.spawn + DistributedDataParallel for true parallel GPU training.
# Launch: python Chess_Brain_mp_spawn_4_12_26.py  (no torchrun needed)

def _handle_training_interrupt(dataset, block_size, move_to_idx, current_lr=None, token_mode='4token'):
    """Handle Ctrl+C during training. Returns dict with 'dataset' and 'lr' keys, or None to quit."""
    print("\n\n" + "=" * 50)
    print("Training interrupted! (Ctrl+C)")
    print("=" * 50)

    result = {'dataset': None, 'lr': None}

    try:
        if current_lr is not None:
            print(f"\nCurrent learning rate: {current_lr:.2e}")
        lr_input = input("New learning rate (or Enter to keep current): ").strip()
        if lr_input:
            try:
                result['lr'] = float(lr_input)
                print(f"Learning rate will change to: {result['lr']:.2e}")
            except ValueError:
                print("Invalid LR, keeping current.")

        choice = input("Load new training data file? (y/n/q=quit): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return None

    if choice == 'q':
        return None

    if choice == 'y':
        print("\nSelect new training data file...")
        file_path = create_file_dialog(
            title="Select New Chess Games File",
            filetypes=[("Chess files", "*.txt *.parquet"), ("Text files", "*.txt"), ("Parquet files", "*.parquet")])
        if not file_path:
            print("No file selected.")
        else:
            if file_path.lower().endswith('.parquet'):
                text = _read_parquet_as_text(file_path)
            else:
                print(f"Loading chess file: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            games = text.split('\n\n')
            games = ['<STARTGAME>' + ' ' + game.strip() + ' ' + '<EOFG>'
                     for game in games if game.strip()]
            text = '\n'.join(games)
            print(f"New dataset loaded. Games: {len(games)}, Characters: {len(text)}")

            if token_mode == 'classic':
                result['dataset'] = ClassicChessMovesDataset(text, block_size, move_to_idx)
            else:
                result['dataset'] = ChessMovesDataset(text, block_size, move_to_idx)
            print(f"New dataset: {len(result['dataset'])} sequences ({token_mode} mode)")

    if result['dataset'] is None and result['lr'] is None:
        print("No changes. Continuing training...\n")
    else:
        print("Continuing training...\n")
    return result


def _ddp_train_worker(rank, world_size, gpu_indices, train_args):
    """DDP worker process — one per GPU. All GPUs train simultaneously via NCCL."""
    try:
        _ddp_setup(rank, world_size)
        torch.cuda.set_device(rank)

        # Per-worker CUDA settings (module-level init was skipped via _CHESS_DDP_WORKER)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # ALL ranks must handle SIGINT to prevent KeyboardInterrupt crashes.
        # Rank 0 catches it and sets a flag; other ranks just ignore it.
        _interrupt_requested = [False]
        _old_sigint = None
        if rank == 0:
            def _sigint_handler(sig, frame):
                _interrupt_requested[0] = True
                print("\n\nCtrl+C detected! Will pause after current batch...")
            _old_sigint = signal.signal(signal.SIGINT, _sigint_handler)
        else:
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Unpack training arguments (prepared by rank 0 before spawn)
        checkpoint_data = train_args['checkpoint_data']
        token_mode = train_args['token_mode']
        model_args = train_args['model_args']
        training_params = train_args['training_params']

        # Pre-tokenized shared-memory tensors (tokenized ONCE before spawn)
        tokens_tensor = train_args['tokens_tensor']
        roles_tensor = train_args.get('roles_tensor')

        n_embd = model_args['n_embd']
        n_head = model_args['n_head']
        n_kv_heads = model_args['n_kv_heads']
        block_size = model_args['block_size']
        n_layer = model_args['n_layer']
        dropout = model_args['dropout']
        vocab_size = model_args['vocab_size']

        batch_size = training_params['batch_size']
        num_epochs = training_params['num_epochs']
        learning_rate = training_params['learning_rate']
        weight_decay = training_params['weight_decay']
        optimizer_choice = training_params['optimizer_choice']
        scheduler_choice = training_params['scheduler_choice']
        start_epoch = training_params['start_epoch']
        start_batch = training_params['start_batch']
        clip_threshold = CHESS_DEFAULTS['max_norm']
        inference_frequency = 500

        # Set globals for this worker process
        global move_to_idx, idx_to_move
        move_to_idx = train_args['move_to_idx']
        idx_to_move = train_args['idx_to_move']

        # Fixed seed ensures all ranks create identical random weights for fresh training.
        # For checkpoint resume this is harmless (weights are overwritten by load_state_dict).
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Create model on this GPU
        model = ChessModel(vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout,
                           use_chess=True, token_mode=token_mode)
        model.start_game_token = move_to_idx['<STARTGAME>']

        # Load checkpoint weights if resuming
        if checkpoint_data is not None:
            state_dict = checkpoint_data['model_state_dict']
            cleaned = {}
            for key, val in state_dict.items():
                new_key = key
                if new_key.startswith('_orig_mod.module.'):
                    new_key = new_key[len('_orig_mod.module.'):]
                elif new_key.startswith('_orig_mod.'):
                    new_key = new_key[len('_orig_mod.'):]
                elif new_key.startswith('module.'):
                    new_key = new_key[len('module.'):]
                cleaned[new_key] = val

            if rank == 0:
                for pname, pval in cleaned.items():
                    if isinstance(pval, torch.Tensor) and pval.is_floating_point() and torch.isnan(pval).any():
                        print(f"\n FATAL: Checkpoint has NaN in '{pname}' -- this checkpoint is corrupted!")
                        return

            model.load_state_dict(cleaned, strict=False)

        model = model.to(rank)

        # Wrap with DDP — NCCL handles all gradient synchronization
        model = DDP(model, device_ids=[rank])

        if rank == 0:
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {num_params:,} parameters on {world_size} GPUs via DDP")
            print(f"Each GPU processes {batch_size // world_size} samples per batch (total batch: {batch_size})")

        # Create optimizer
        if optimizer_choice == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                betas=(0.9, 0.999), eps=1e-8, fused=True
            )
        else:
            optimizer = Adafactor(
                model.parameters(), lr=learning_rate, scale_parameter=True,
                relative_step=False, warmup_init=False, clip_threshold=clip_threshold,
                weight_decay=weight_decay, beta1=0.9, eps=(1e-30, 1e-3)
            )

        # Load optimizer state from checkpoint (only need one copy — all ranks are identical)
        if checkpoint_data is not None:
            opt_state = checkpoint_data.get('optimizer_state_dict')
            if opt_state is not None:
                if isinstance(opt_state, list):
                    opt_state = opt_state[0]
                try:
                    optimizer.load_state_dict(opt_state)
                    if rank == 0:
                        print("Optimizer state loaded from checkpoint")
                except ValueError as e:
                    if rank == 0:
                        print(f"Optimizer state incompatible, using fresh state: {e}")

            # Apply user-requested learning rate
            optimizer.param_groups[0]['lr'] = learning_rate
            if rank == 0:
                print(f"Learning rate set to: {learning_rate}")

        # Scheduler
        if scheduler_choice == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=100, threshold=0.001, min_lr=1e-7)
        elif scheduler_choice == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Dataset from pre-tokenized shared-memory tensors (no re-tokenization)
        dataset = _PreTokenizedDataset(tokens_tensor, roles_tensor, block_size, token_mode, move_to_idx)

        if rank == 0:
            print(f"Dataset: {len(dataset)} sequences ({token_mode} mode)")

        # Auto-disable gradient checkpointing if memory headroom > 20%
        try:
            total_mem = torch.cuda.get_device_properties(rank).total_mem
            used_mem = torch.cuda.memory_allocated(rank)
            headroom = 1.0 - (used_mem / total_mem)
            if headroom > 0.20:
                if rank == 0:
                    print(f"Memory headroom {headroom:.0%} > 20% -- disabling gradient checkpointing for speed")
                model.module.set_gradient_checkpointing(False)
        except Exception:
            pass

        # Training loop
        model.train()
        running_loss = 0.0
        total_batches = 0
        epoch_losses = []
        all_text = ""

        # Plateau detection state
        batch_group_losses = []
        plateau_check_counter = 0
        previous_plateau_avg = None

        epoch = start_epoch
        while epoch < num_epochs:
            epoch_loss = 0.0
            epoch_batches = 0

            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            sampler.set_epoch(epoch)

            num_workers = min(8, os.cpu_count() // max(world_size, 1))
            data_loader = DataLoader(
                dataset, batch_size=batch_size // world_size, sampler=sampler,
                drop_last=True, num_workers=num_workers,
                pin_memory=True, persistent_workers=(num_workers > 0),
                prefetch_factor=4 if num_workers > 0 else None
            )

            if rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, {len(data_loader)} batches per GPU")

            _batch_interrupted = False
            _quit_training = False
            _new_data = False

            for batch_idx, (x, y, y_roles) in enumerate(data_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue

                x = x.to(rank, non_blocking=True)
                y = y.to(rank, non_blocking=True)
                y_roles = y_roles.to(rank, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    output, loss = model(x, targets=y, target_roles=y_roles)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
                optimizer.step()

                loss_val = loss.item()
                running_loss += loss_val
                epoch_loss += loss_val
                total_batches += 1
                epoch_batches += 1

                # Progress reporting — rank 0 only
                if rank == 0 and (batch_idx + 1) % inference_frequency == 0:
                    avg_loss = running_loss / total_batches
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], "
                          f"Avg Loss: {avg_loss:.4f} (DDP {world_size} GPUs)")

                    # Per-head loss diagnostics
                    model_raw = model.module
                    if hasattr(model_raw, '_last_head_losses'):
                        hl = model_raw._last_head_losses
                        print(f"  Head losses -> color: {hl['color']:.4f}  from: {hl['from']:.4f}  "
                              f"to: {hl['to']:.4f}  promo: {hl['promo']:.4f}")

                    # Plateau detection
                    if scheduler_choice == 'plateau':
                        batch_group_losses.append(avg_loss)
                        plateau_check_counter += 1
                        if len(batch_group_losses) > 20:
                            batch_group_losses = batch_group_losses[-20:]

                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"LR: {current_lr:.2e} | Loss history: {len(batch_group_losses)} samples")

                        if plateau_check_counter >= 10 and len(batch_group_losses) >= 20:
                            plateau_check_counter = 0
                            current_avg = sum(batch_group_losses) / len(batch_group_losses)
                            if previous_plateau_avg is not None:
                                improvement = previous_plateau_avg - current_avg
                                if improvement >= 0.001:
                                    print(f"Improvement: {improvement:.4f} (>= 0.001) - continuing at LR {current_lr:.2e}")
                                else:
                                    print(f"Plateau: {improvement:.4f} < 0.001 - reducing LR from {current_lr:.2e}")
                                    scheduler.step(float('inf'))
                                    new_lr = optimizer.param_groups[0]['lr']
                                    print(f"   New LR: {new_lr:.2e}")
                            previous_plateau_avg = current_avg

                    # Generate sample moves
                    all_text = test_progress(
                        epoch, num_epochs, batch_idx, data_loader, avg_loss,
                        model.module, x, 50, all_text, idx_to_move
                    )

                    # Save checkpoint — rank 0 only, using model.module (unwrapped)
                    save_model_all(
                        model.module, all_text, n_embd, n_head, n_kv_heads, n_layer, dropout,
                        block_size, epoch, batch_idx, batch_size, optimizer, scheduler, None,
                        avg_loss, learning_rate, weight_decay, gpu_indices, token_mode
                    )

                    running_loss = 0.0
                    total_batches = 0
                    torch.cuda.empty_cache()

                # Ctrl+C handling — rank 0 checks, then broadcasts decision to all ranks.
                # This is a COLLECTIVE operation: all ranks must participate in the broadcast.
                interrupt_tensor = torch.zeros(1, dtype=torch.long, device=rank)
                if rank == 0 and _interrupt_requested[0]:
                    _interrupt_requested[0] = False
                    interrupt_tensor[0] = 1
                dist.broadcast(interrupt_tensor, src=0)

                if interrupt_tensor[0] == 1:
                    # Only rank 0 shows the interactive menu.
                    # Ranks 1-3 block at the broadcasts below, waiting for rank 0's decision.
                    # Actions: 0=quit, 1=LR change, 2=no change, 3=new data
                    action_tensor = torch.zeros(1, dtype=torch.long, device=rank)
                    lr_tensor = torch.zeros(1, dtype=torch.float64, device=rank)

                    if rank == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"\nCurrent learning rate: {current_lr:.2e}")
                        try:
                            lr_input = input("New learning rate (or Enter to keep current): ").strip()
                            if lr_input:
                                try:
                                    new_lr_val = float(lr_input)
                                    lr_tensor[0] = new_lr_val
                                    action_tensor[0] = 1  # LR change
                                    print(f"Learning rate will change to: {new_lr_val:.2e}")
                                except ValueError:
                                    print("Invalid LR, keeping current.")

                            choice = input("Load new data / Quit / Continue? (d=new data, q=quit, Enter=continue): ").strip().lower()
                            if choice == 'q':
                                action_tensor[0] = 0
                            elif choice == 'd':
                                action_tensor[0] = 3  # new data
                            elif action_tensor[0] != 1:
                                action_tensor[0] = 2  # no change
                        except (KeyboardInterrupt, EOFError):
                            action_tensor[0] = 0

                    dist.broadcast(action_tensor, src=0)
                    dist.broadcast(lr_tensor, src=0)
                    action = action_tensor[0].item()

                    if action == 0:
                        if rank == 0:
                            avg_loss = running_loss / max(total_batches, 1)
                            save_model_all(
                                model.module, all_text, n_embd, n_head, n_kv_heads, n_layer, dropout,
                                block_size, epoch, batch_idx, batch_size, optimizer, scheduler, None,
                                avg_loss, learning_rate, weight_decay, gpu_indices, token_mode
                            )
                        _quit_training = True
                        _batch_interrupted = True
                        break
                    elif action == 1:
                        new_lr = lr_tensor[0].item()
                        for pg in optimizer.param_groups:
                            pg['lr'] = new_lr
                        if rank == 0:
                            print(f"Learning rate updated to {new_lr:.2e} on all {world_size} GPUs\n")
                        _batch_interrupted = True
                        break
                    elif action == 3:
                        # New data: rank 0 loads + tokenizes, saves to temp file,
                        # all ranks load from the same file (same machine = same filesystem).
                        import tempfile
                        _tmp_path = os.path.join(tempfile.gettempdir(), '_chess_ddp_new_data.pt')
                        data_ready = torch.zeros(1, dtype=torch.long, device=rank)

                        if rank == 0:
                            print("\nSelect new training data file...")
                            file_path = create_file_dialog(
                                title="Select New Chess Games File",
                                filetypes=[("Chess files", "*.txt *.parquet"), ("Text files", "*.txt"), ("Parquet files", "*.parquet")])
                            if file_path:
                                if file_path.lower().endswith('.parquet'):
                                    new_text = _read_parquet_as_text(file_path)
                                else:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        new_text = f.read()
                                games = new_text.split('\n\n')
                                games = ['<STARTGAME> ' + g.strip() + ' <EOFG>' for g in games if g.strip()]
                                new_text = '\n'.join(games)
                                print(f"Tokenizing new data ({len(games)} games)...")
                                if token_mode == 'classic':
                                    tmp_ds = ClassicChessMovesDataset(new_text, block_size, move_to_idx)
                                    torch.save({'tokens': tmp_ds.tokens_tensor, 'roles': None}, _tmp_path)
                                else:
                                    tmp_ds = ChessMovesDataset(new_text, block_size, move_to_idx)
                                    torch.save({'tokens': tmp_ds.tokens_tensor, 'roles': tmp_ds.roles_tensor}, _tmp_path)
                                del tmp_ds, new_text
                                data_ready[0] = 1
                                print(f"New data ready for all GPUs.")
                            else:
                                print("No file selected. Continuing training...")

                        # Broadcast whether data was actually loaded (user might cancel dialog)
                        dist.broadcast(data_ready, src=0)

                        if data_ready[0] == 1:
                            # All ranks load from the same temp file
                            new_data = torch.load(_tmp_path, map_location='cpu', weights_only=True)
                            dataset = _PreTokenizedDataset(
                                new_data['tokens'], new_data.get('roles'),
                                block_size, token_mode, move_to_idx
                            )
                            del new_data

                            # Wait for ALL ranks to finish loading before rank 0 deletes the file
                            dist.barrier()
                            if rank == 0:
                                os.remove(_tmp_path)
                                print(f"New dataset: {len(dataset)} sequences. Restarting from epoch 1.\n")

                            # Also apply LR change if user entered one before selecting 'd'
                            new_lr = lr_tensor[0].item()
                            if new_lr > 0:
                                for pg in optimizer.param_groups:
                                    pg['lr'] = new_lr
                                if rank == 0:
                                    print(f"Learning rate updated to {new_lr:.2e} on all {world_size} GPUs")

                            _new_data = True
                            _batch_interrupted = True
                            break
                        # else: user cancelled, continue training
                    # action == 2: no change, continue
                    if rank == 0:
                        print("No changes. Continuing training...\n")

                del output, loss, x, y, y_roles

            # Handle interrupt after batch loop
            if _batch_interrupted:
                if _quit_training:
                    break
                elif _new_data:
                    epoch = 0
                    start_epoch = 0
                    start_batch = 0
                    epoch_losses = []
                    running_loss = 0.0
                    total_batches = 0
                    continue
                else:
                    continue

            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            epoch_losses.append(avg_epoch_loss)
            if rank == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

            if scheduler_choice == 'exponential':
                scheduler.step()
                if rank == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Exponential LR decay: {current_lr:.2e}")

            epoch += 1

        # Normal completion
        if rank == 0 and _old_sigint is not None:
            signal.signal(signal.SIGINT, _old_sigint)

        if rank == 0 and epoch_losses:
            initial_loss = epoch_losses[0]
            final_loss = epoch_losses[-1]
            total_improvement = initial_loss - final_loss
            print(f"\nTraining Summary:")
            print(f"Initial loss: {initial_loss:.4f}")
            print(f"Final loss: {final_loss:.4f}")
            print(f"Total improvement: {total_improvement:.4f}")
            print(f"Improvement rate: {total_improvement/len(epoch_losses):.4f} per epoch")
            print("Training completed!")

    finally:
        _ddp_cleanup()


def _train_chess_model_core(text, checkpoint_data=None, token_mode='4token'):
    """
    Core training logic. For multi-GPU: launches DDP workers via mp.spawn.
    For single GPU: trains directly (no spawn overhead).
    """
    print("ChessBrain - Chess Move Prediction LLM (DDP Multi-GPU)")
    print("=" * 50)

    vocab_size = len(move_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    global gpu_indices
    if 'gpu_indices' not in globals() or gpu_indices is None:
        gpu_indices = [0] if torch.cuda.is_available() else None

    learning_rate = CHESS_DEFAULTS['learning_rate']
    weight_decay = CHESS_DEFAULTS['weight_decay']

    if checkpoint_data:
        model, vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, \
        optimizer_state_dict, scheduler_state_dict, scaler_state_dict, start_epoch, start_batch, checkpoint_hyperparams = checkpoint_data
        model.start_game_token = move_to_idx['<STARTGAME>']
        print(f"Resuming from checkpoint: epoch {start_epoch}, batch {start_batch}")
        print(f"Model architecture: {n_layer} layers, {n_head} heads, {n_embd} embedding dim")
        print(f"Block size: {block_size}, Vocab size: {vocab_size}")

        saved_gpu_indices = checkpoint_hyperparams.get('gpu_indices')
        if saved_gpu_indices:
            print(f"Checkpoint originally trained on GPUs: {saved_gpu_indices}")
            print(f"System has {torch.cuda.device_count()} GPUs available")

        # For DDP, any GPU count works — optimizer state is a single copy
        # Allow user to select GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            default_gpus = ",".join(str(i) for i in range(num_gpus))
            custom_gpus = input(f"Enter GPU indices (default: {default_gpus}): ").strip()
            if custom_gpus:
                try:
                    gpu_indices = [int(idx.strip()) for idx in custom_gpus.split(',')]
                except ValueError:
                    gpu_indices = list(range(num_gpus))
            else:
                gpu_indices = list(range(num_gpus))
        else:
            gpu_indices = [0]
        print(f"Using GPUs: {gpu_indices}")

        batch_size = enter_batch_size(n_embd, n_head, block_size, n_layer,
                                      checkpoint_hyperparams.get('batch_size', 256),
                                      gpu_indices, n_kv_heads)
        num_epochs = int(get_input_with_default("Number of epochs", 20))

        if optimizer_state_dict:
            if isinstance(optimizer_state_dict, list):
                saved_learning_rate = optimizer_state_dict[0]['param_groups'][0]['lr']
            else:
                saved_learning_rate = optimizer_state_dict['param_groups'][0]['lr']
        else:
            saved_learning_rate = CHESS_DEFAULTS['learning_rate']
        print(f"Current learning rate from checkpoint: {saved_learning_rate}")
        learning_rate = float(get_input_with_default("Learning rate", saved_learning_rate))
        weight_decay = float(get_input_with_default("Weight decay",
                                                     checkpoint_hyperparams.get('weight_decay', CHESS_DEFAULTS['weight_decay'])))
        dropout = float(get_input_with_default("Dropout", checkpoint_hyperparams['dropout']))

        optimizer_choice = "adamw"
        scheduler_choice = get_input_with_default("Scheduler (cosine/plateau/exponential)", "plateau").lower()

        # Prepare checkpoint data dict for worker processes
        # MUST be CPU tensors — mp.spawn can't serialize CUDA tensors across processes
        def _to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu()
            elif isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_to_cpu(v) for v in obj]
            return obj

        cpu_state = _to_cpu(model.state_dict())
        cpu_opt = optimizer_state_dict
        if isinstance(cpu_opt, list):
            cpu_opt = cpu_opt[0]
        cpu_opt = _to_cpu(cpu_opt)
        checkpoint_dict = {
            'model_state_dict': cpu_state,
            'optimizer_state_dict': cpu_opt,
        }
        # Free the CUDA model — workers will create their own
        del model
        torch.cuda.empty_cache()
    else:
        start_epoch = 0
        start_batch = 0
        checkpoint_dict = None

        n_embd = int(get_input_with_default("Embedding dimensions", CHESS_DEFAULTS['n_embd']))
        n_head = int(get_input_with_default("Number of query heads", CHESS_DEFAULTS['n_head']))
        n_kv_heads = int(get_input_with_default("Number of KV heads", CHESS_DEFAULTS['n_kv_heads']))
        block_size = int(get_input_with_default("Sequence length",
                                                 128 if token_mode == 'classic' else CHESS_DEFAULTS['block_size']))
        n_layer = int(get_input_with_default("Number of layers", CHESS_DEFAULTS['n_layer']))
        dropout = float(get_input_with_default("Dropout", CHESS_DEFAULTS['dropout']))
        num_epochs = int(get_input_with_default("Number of epochs", CHESS_DEFAULTS['num_epochs']))

        if n_embd % n_head != 0:
            original_embd = n_embd
            while n_embd % n_head != 0:
                n_embd += n_head
            print(f"Adjusted n_embd from {original_embd} to {n_embd}")

        vocab_size = len(move_to_idx)

        # GPU selection
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            default_gpus = ",".join(str(i) for i in range(num_gpus))
            custom_gpus = input(f"Enter GPU indices (default: {default_gpus}): ").strip()
            if custom_gpus:
                try:
                    gpu_indices = [int(idx.strip()) for idx in custom_gpus.split(',')]
                except ValueError:
                    gpu_indices = list(range(num_gpus))
            else:
                gpu_indices = list(range(num_gpus))
        else:
            gpu_indices = [0]
        print(f"Using GPUs: {gpu_indices}")

        batch_size = enter_batch_size(n_embd, n_head, block_size, n_layer,
                                      CHESS_DEFAULTS['batch_size'], gpu_indices, n_kv_heads)

        optimizer_choice = get_input_with_default("Optimizer (adamw/adfactor)", "adamw").lower()
        scheduler_choice = get_input_with_default("Scheduler (cosine/plateau/exponential)", "plateau").lower()

    # Ensure batch_size divisible by GPU count
    world_size = len(gpu_indices)
    if batch_size % world_size != 0:
        batch_size = (batch_size // world_size) * world_size
        print(f"Adjusted batch size to {batch_size} (divisible by {world_size} GPUs)")

    # Tokenize ONCE here — avoids re-tokenizing in every DDP worker process
    print("Tokenizing data (once, before launching GPU workers)...")
    if token_mode == 'classic':
        _temp_dataset = ClassicChessMovesDataset(text, block_size, move_to_idx)
        tokens_tensor = _temp_dataset.tokens_tensor
        roles_tensor = None
    else:
        _temp_dataset = ChessMovesDataset(text, block_size, move_to_idx)
        tokens_tensor = _temp_dataset.tokens_tensor
        roles_tensor = _temp_dataset.roles_tensor
    num_games = len(_temp_dataset)
    del _temp_dataset, text
    print(f"Tokenized: {len(tokens_tensor):,} tokens, {num_games:,} games")

    # Put tensors in shared memory so all DDP workers access the same data (no copies)
    tokens_tensor.share_memory_()
    if roles_tensor is not None:
        roles_tensor.share_memory_()

    model_args = {
        'n_embd': n_embd, 'n_head': n_head, 'n_kv_heads': n_kv_heads,
        'block_size': block_size, 'n_layer': n_layer, 'dropout': dropout,
        'vocab_size': vocab_size,
    }
    training_params = {
        'batch_size': batch_size, 'num_epochs': num_epochs,
        'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'optimizer_choice': optimizer_choice, 'scheduler_choice': scheduler_choice,
        'start_epoch': start_epoch, 'start_batch': start_batch,
    }
    train_args = {
        'tokens_tensor': tokens_tensor,
        'roles_tensor': roles_tensor,
        'checkpoint_data': checkpoint_dict,
        'token_mode': token_mode,
        'model_args': model_args,
        'training_params': training_params,
        'move_to_idx': move_to_idx,
        'idx_to_move': idx_to_move,
    }

    if world_size > 1:
        print(f"\nLaunching DDP training on {world_size} GPUs: {gpu_indices}")
        print(f"All {world_size} GPUs will train SIMULTANEOUSLY via NCCL")
        print(f"Expected ~{world_size}x speedup over sequential GPU usage\n")

        # Set CUDA_VISIBLE_DEVICES so rank 0,1,2... map to the selected physical GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_indices))

        # Tell child processes to skip module-level CUDA init (prevents 444 MiB
        # parasitic context per process on GPU 0)
        os.environ['_CHESS_DDP_WORKER'] = '1'

        mp.spawn(
            _ddp_train_worker,
            args=(world_size, gpu_indices, train_args),
            nprocs=world_size,
            join=True
        )

        os.environ.pop('_CHESS_DDP_WORKER', None)
    else:
        # Single GPU — train directly without DDP overhead
        print(f"\nSingle GPU training on GPU {gpu_indices[0]}")
        _single_gpu_train(gpu_indices[0], train_args)


def _single_gpu_train(gpu_id, train_args):
    """Single-GPU training path — no DDP, no spawn. Identical logic to the original."""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    checkpoint_data = train_args['checkpoint_data']
    token_mode = train_args['token_mode']
    model_args = train_args['model_args']
    training_params = train_args['training_params']
    tokens_tensor = train_args['tokens_tensor']
    roles_tensor = train_args.get('roles_tensor')

    n_embd = model_args['n_embd']
    n_head = model_args['n_head']
    n_kv_heads = model_args['n_kv_heads']
    block_size = model_args['block_size']
    n_layer = model_args['n_layer']
    dropout = model_args['dropout']
    vocab_size = model_args['vocab_size']

    batch_size = training_params['batch_size']
    num_epochs = training_params['num_epochs']
    learning_rate = training_params['learning_rate']
    weight_decay = training_params['weight_decay']
    optimizer_choice = training_params['optimizer_choice']
    scheduler_choice = training_params['scheduler_choice']
    start_epoch = training_params['start_epoch']
    start_batch = training_params['start_batch']
    clip_threshold = CHESS_DEFAULTS['max_norm']
    inference_frequency = 500

    global move_to_idx, idx_to_move, gpu_indices
    move_to_idx = train_args['move_to_idx']
    idx_to_move = train_args['idx_to_move']

    # Create model
    model = ChessModel(vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout,
                       use_chess=True, token_mode=token_mode)
    model.start_game_token = move_to_idx['<STARTGAME>']

    if checkpoint_data is not None:
        state_dict = checkpoint_data['model_state_dict']
        cleaned = {}
        for key, val in state_dict.items():
            new_key = key
            if new_key.startswith('_orig_mod.module.'):
                new_key = new_key[len('_orig_mod.module.'):]
            elif new_key.startswith('_orig_mod.'):
                new_key = new_key[len('_orig_mod.'):]
            elif new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
            cleaned[new_key] = val

        for pname, pval in cleaned.items():
            if isinstance(pval, torch.Tensor) and pval.is_floating_point() and torch.isnan(pval).any():
                print(f"\n FATAL: Checkpoint has NaN in '{pname}' -- corrupted!")
                return
        model.load_state_dict(cleaned, strict=False)

    model = model.to(device)

    # torch.compile
    if not os.environ.get('CHESS_NO_COMPILE') and hasattr(torch, 'compile'):
        try:
            print("Enabling torch.compile()...")
            model = torch.compile(model, mode='reduce-overhead')
            print("torch.compile() enabled")
        except Exception as e:
            print(f"torch.compile() failed: {e}, continuing without")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters, single GPU")

    # Auto-disable gradient checkpointing
    try:
        total_mem = torch.cuda.get_device_properties(gpu_id).total_mem
        used_mem = torch.cuda.memory_allocated(gpu_id)
        headroom = 1.0 - (used_mem / total_mem)
        if headroom > 0.20:
            print(f"Memory headroom {headroom:.0%} -- disabling gradient checkpointing")
            get_model_module(model).set_gradient_checkpointing(False)
    except Exception:
        pass

    # Optimizer
    model_params = get_model_module(model).parameters()
    if optimizer_choice == 'adamw':
        optimizer = torch.optim.AdamW(
            model_params, lr=learning_rate, weight_decay=weight_decay,
            betas=(0.9, 0.999), eps=1e-8, fused=True
        )
    else:
        optimizer = Adafactor(
            model_params, lr=learning_rate, scale_parameter=True,
            relative_step=False, warmup_init=False, clip_threshold=clip_threshold,
            weight_decay=weight_decay, beta1=0.9, eps=(1e-30, 1e-3)
        )

    if checkpoint_data is not None:
        opt_state = checkpoint_data.get('optimizer_state_dict')
        if opt_state is not None:
            if isinstance(opt_state, list):
                opt_state = opt_state[0]
            try:
                optimizer.load_state_dict(opt_state)
                print("Optimizer state loaded from checkpoint")
            except ValueError as e:
                print(f"Optimizer state incompatible: {e}")
        optimizer.param_groups[0]['lr'] = learning_rate

    # Scheduler
    if scheduler_choice == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=100, threshold=0.001, min_lr=1e-7)
    elif scheduler_choice == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    scaler = GradScaler()

    # Dataset from pre-tokenized tensors (tokenized once in _train_chess_model_core)
    dataset = _PreTokenizedDataset(tokens_tensor, roles_tensor, block_size, token_mode, move_to_idx)
    print(f"Dataset: {len(dataset)} sequences ({token_mode} mode)")

    # Ctrl+C handler
    _interrupt_requested = [False]
    def _sigint_handler(sig, frame):
        _interrupt_requested[0] = True
        print("\n\nCtrl+C detected! Will pause after current batch...")
    _old_sigint = signal.signal(signal.SIGINT, _sigint_handler)

    # Training loop
    model.train()
    running_loss = 0.0
    total_batches = 0
    epoch_losses = []
    all_text = ""
    is_blackwell_gpu = torch.cuda.get_device_capability(gpu_id) == (12, 1)

    batch_group_losses = []
    plateau_check_counter = 0
    previous_plateau_avg = None

    num_workers = min(8, os.cpu_count() // 2)

    epoch = start_epoch
    while epoch < num_epochs:
        epoch_loss = 0.0
        epoch_batches = 0

        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None
        )
        print(f"Epoch {epoch+1}/{num_epochs}, {len(data_loader)} batches")

        _batch_interrupted = False
        _new_data = False
        _quit_training = False

        for batch_idx, (x, y, y_roles) in enumerate(data_loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_roles = y_roles.to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                output, loss = model(x, targets=y, target_roles=y_roles)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            model_params_list = list(get_model_module(model).parameters())
            total_norm = torch.nn.utils.clip_grad_norm_(model_params_list, clip_threshold)
            if total_norm > clip_threshold and batch_idx % 1000 == 0:
                print(f"Gradient clipping at batch {batch_idx}. Norm: {total_norm:.2f}")

            optimizer.step()

            loss_val = loss.item()
            running_loss += loss_val
            epoch_loss += loss_val
            total_batches += 1
            epoch_batches += 1

            current_batch_x = x

            del output, loss, x, y, y_roles
            if is_blackwell_gpu:
                torch.cuda.empty_cache()

            if (batch_idx + 1) % inference_frequency == 0:
                avg_loss = running_loss / total_batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Avg Loss: {avg_loss:.4f}")

                if scheduler_choice == 'plateau':
                    batch_group_losses.append(avg_loss)
                    plateau_check_counter += 1
                    if len(batch_group_losses) > 20:
                        batch_group_losses = batch_group_losses[-20:]

                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"LR: {current_lr:.2e} | Loss history: {len(batch_group_losses)} samples")

                    if plateau_check_counter >= 10 and len(batch_group_losses) >= 20:
                        plateau_check_counter = 0
                        current_avg = sum(batch_group_losses) / len(batch_group_losses)
                        if previous_plateau_avg is not None:
                            improvement = previous_plateau_avg - current_avg
                            if improvement >= 0.001:
                                print(f"Improvement: {improvement:.4f} - continuing at LR {current_lr:.2e}")
                            else:
                                print(f"Plateau: {improvement:.4f} < 0.001 - reducing LR")
                                scheduler.step(float('inf'))
                                print(f"   New LR: {optimizer.param_groups[0]['lr']:.2e}")
                        previous_plateau_avg = current_avg

                running_loss = 0.0
                total_batches = 0

                all_text = test_progress(
                    epoch, num_epochs, batch_idx, data_loader, loss_val,
                    model, current_batch_x, 50, all_text, idx_to_move
                )

                save_model_all(
                    model, all_text, n_embd, n_head, n_kv_heads, n_layer, dropout,
                    block_size, epoch, batch_idx, batch_size, optimizer, scheduler, scaler,
                    loss_val, learning_rate, weight_decay, gpu_indices, token_mode
                )

                if is_blackwell_gpu:
                    torch.cuda.empty_cache()

            if _interrupt_requested[0]:
                _interrupt_requested[0] = False
                current_lr = optimizer.param_groups[0]['lr']
                result = _handle_training_interrupt(dataset, block_size, move_to_idx, current_lr, token_mode)
                if result is None:
                    _batch_interrupted = True
                    _quit_training = True
                    break
                if result['lr'] is not None:
                    for pg in optimizer.param_groups:
                        pg['lr'] = result['lr']
                    print(f"Learning rate updated to {result['lr']:.2e}")
                if result['dataset'] is not None:
                    dataset = result['dataset']
                    _new_data = True
                _batch_interrupted = True
                break

        if _batch_interrupted:
            if _quit_training:
                break
            elif _new_data:
                epoch = 0
                start_epoch = 0
                start_batch = 0
                epoch_losses = []
                running_loss = 0.0
                total_batches = 0
                continue
            else:
                continue

        avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

        if scheduler_choice == 'exponential':
            scheduler.step()
            print(f"Exponential LR decay: {optimizer.param_groups[0]['lr']:.2e}")

        if len(epoch_losses) > 1:
            loss_change = epoch_losses[-2] - epoch_losses[-1]
            print(f"Loss change from previous epoch: {loss_change:+.4f}")

        epoch += 1

    signal.signal(signal.SIGINT, _old_sigint)

    if epoch_losses:
        print(f"\nTraining Summary:")
        print(f"Initial loss: {epoch_losses[0]:.4f}")
        print(f"Final loss: {epoch_losses[-1]:.4f}")
        print(f"Improvement: {epoch_losses[0] - epoch_losses[-1]:.4f}")
    print("Training completed!")


def train_chess_model():
    """Main training entry point."""
    text, checkpoint_data, token_mode = load_data_interactive()
    _train_chess_model_core(text, checkpoint_data, token_mode)


def load_data_interactive():
    """Load data in single process mode (interactive).
    Returns (text, checkpoint_data, token_mode).
    """
    global gpu_indices, move_to_idx, idx_to_move
    text = ""
    checkpoint_data = None
    token_mode = 'classic'

    load_or_create = get_input_with_default("Load a model file or Create a new model? (l/c)", "c").lower()

    if load_or_create == 'l':
        loaded_data = load_model_file()
        if loaded_data:
            checkpoint_data = loaded_data
            checkpoint_hyperparams = loaded_data[-1]
            token_mode = checkpoint_hyperparams.get('token_mode', '4token')
            print(f"Checkpoint token mode: {token_mode}")
        else:
            print("Failed to load model. Creating a new one.")
            checkpoint_data = None
    else:
        checkpoint_data = None

    if checkpoint_data is None:
        print("\nToken Mode Selection:")
        print("  classic  - 1 token per move (~20K vocab, 512 moves of context)")
        print("  4token   - 4 tokens per ply (140 vocab, role-specific heads)")
        token_mode = get_input_with_default("Token mode (classic/4token)", "classic").lower()
        if token_mode not in ('classic', '4token'):
            print(f"Unknown mode '{token_mode}', defaulting to 'classic'")
            token_mode = 'classic'

        if token_mode == 'classic':
            move_to_idx = create_classic_move_to_idx()
            idx_to_move = create_classic_idx_to_move(move_to_idx)
            print(f"Classic tokenizer created: {len(move_to_idx)} tokens")
        else:
            move_to_idx = create_move_to_idx()
            idx_to_move = create_idx_to_move()
            print(f"4-token tokenizer created: {len(move_to_idx)} tokens")

    # Always use interactive file selection
    file_path = create_file_dialog(
        title="Select Chess Games File for Training",
        filetypes=[("Chess files", "*.txt *.parquet"), ("Text files", "*.txt"), ("Parquet files", "*.parquet")])
    if not file_path:
        print("No chess file selected. Exiting.")
        exit()

    if file_path.lower().endswith('.parquet'):
        text = _read_parquet_as_text(file_path)
    else:
        print(f"Loading chess file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    games = text.split('\n\n')
    games = ['<STARTGAME>' + ' ' + game.strip() + ' ' + '<EOFG>' for game in games if game.strip()]
    text = '\n'.join(games)
    print(f"Chess dataset loaded. Total games: {len(games)}, Total characters: {len(text)}")

    return text, checkpoint_data, token_mode


if __name__ == "__main__":
    # mp.spawn already uses spawn start method internally for DDP workers.
    # Do NOT set it globally — that would force DataLoader workers to also use spawn,
    # causing each one to re-import the module and create a 444 MiB CUDA context on GPU 0.
    # DataLoader workers use fork (Linux default), which shares memory without re-import.

    print("ChessBrain - Chess Move Prediction LLM (DDP Multi-GPU)")
    print("=" * 50)
    print("Single process mode - interactive GUI setup")
    train_chess_model()
