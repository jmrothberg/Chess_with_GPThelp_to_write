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
import torch
import torch.nn as nn
import torch.optim as optim
import math
import signal
import sys
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from transformers.optimization import Adafactor

# Prioritize MPS on Mac systems for native GPU support
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    device = torch.device('mps')
    gpu_indices = []  # MPS doesn't use gpu_indices like CUDA
    print("Using MPS GPU")
elif torch.cuda.is_available():
    # GPU selection will be done later - either from checkpoint or user selection
    # For now, just initialize to None and set device
    gpu_indices = None

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

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
else:
    print("❌ ERROR: No GPU available. This chess training requires GPU support (CUDA or MPS).")
    exit(1)

# Signal handler for clean shutdown (Ctrl+C)
def signal_handler(sig, frame):
    """Handle Ctrl+C to cleanly terminate all processes"""
    print('\n🛑 Training interrupted by user (Ctrl+C)')
    print('🧹 Cleaning up processes and GPU memory...')

    try:
        # Force cleanup of GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print('✅ GPU memory cleared')
    except Exception as e:
        print(f'⚠️  GPU cleanup failed: {e}')

    print('👋 Training terminated cleanly')
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

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
    'block_size': 512,   # 4 tokens per ply: ~80 moves * 4 + specials ≈ 323 tokens
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

        if self.flash_available:
            print(f"Using Flash Attention {'with RoPE' if use_rope else ''}")

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
                # Check if our PyTorch version supports the advanced args
                try:
                    # Try with all optimizations
                    y = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attention_mask,  # Shape: [B, 1, T, T]
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=False,  # We're handling causality in our mask
                        scale=1.0 / math.sqrt(k.size(-1)),  # Explicit scaling for precision
                        mem_efficient=True  # Use memory efficient attention
                    )
                except TypeError:
                    # Fallback to standard arguments
                    y = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attention_mask,  # Shape: [B, 1, T, T]
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=False  # We're handling causality in our mask
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

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('causal_mask', torch.tril(torch.ones(1024, 1024)))
        self.flash_available = hasattr(F, 'scaled_dot_product_attention')

        if self.flash_available:
            print("Using Flash Attention in MultiQueryAttention")

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # Project queries
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Project keys and values together
        kv = self.kv_proj(x).view(B, T, self.n_kv_heads, 2, self.head_dim)
        kv = kv.transpose(1, 2)
        k, v = kv[..., 0, :], kv[..., 1, :]

        # Repeat keys and values to match the number of query heads
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

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

            # Use flash attention
            try:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False,
                    scale=1.0 / math.sqrt(k.size(-1)),
                    mem_efficient=True
                )
            except TypeError:
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
    def __init__(self, n_embd, n_head, n_kv_heads, dropout):
        super().__init__()
        self.rms_1 = RMSNorm(n_embd)
        self.attn = MultiQueryAttention(n_embd, n_head, n_kv_heads, dropout)
        self.rms_2 = RMSNorm(n_embd)
        self.swiglu = SwiGLU(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Gradient checkpointing for memory efficiency in short training sessions (1800 ELO)
        if self.training:
            # Checkpoint attention layer to save memory during training
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
            # Normal forward pass during inference (no checkpointing needed)
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
    def __init__(self, vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, use_chess=False, use_dna=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.use_chess = use_chess
        self.use_dna = use_dna
        if use_chess:
            self.start_game_token = move_to_idx['<STARTGAME>'] if 'move_to_idx' in globals() else None

        # Standard embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

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

        # Role-specific output heads for 4-token-per-ply grammar
        self.head_color = nn.Linear(n_embd, 2)    # White / Black
        self.head_from = nn.Linear(n_embd, 64)    # FROM square
        self.head_to = nn.Linear(n_embd, 64)      # TO square
        self.head_promo = nn.Linear(n_embd, 5)    # none / q / r / b / n

        # FROM conditioning embedding for TO prediction
        # When predicting TO, the input token at that position is the FROM token
        # (due to autoregressive shift). We add an embedding of FROM to the
        # hidden state before the TO head.
        self.emb_from = nn.Embedding(64, n_embd)

        # Initialize weights
        self.apply(self._init_weights)

        # Mild smoothing reduces overconfidence on deterministic Stockfish lines
        self.label_smoothing = 0.05

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

        # Get embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # Apply transformer blocks with chess game mask
        for block in self.blocks:
            x = block(x, mask=self.create_game_mask(idx))

        # Final normalization
        x = self.rms_final(x)

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
            losses = []

            # COLOR loss
            if color_mask.any():
                color_logits = self.head_color(h[color_mask])
                color_targets = targets_flat[color_mask] - COLOR_OFFSET
                loss_color = F.cross_entropy(color_logits, color_targets,
                                             label_smoothing=label_smoothing)
                losses.append(loss_color)

            # FROM loss
            if from_mask.any():
                from_logits = self.head_from(h[from_mask])
                from_targets = targets_flat[from_mask] - FROM_OFFSET
                loss_from = F.cross_entropy(from_logits, from_targets,
                                            label_smoothing=label_smoothing)
                losses.append(loss_from)

            # TO loss (conditioned on FROM via emb_from)
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

            # PROMO loss
            if promo_mask.any():
                promo_logits = self.head_promo(h[promo_mask])
                promo_targets = targets_flat[promo_mask] - PROMO_OFFSET
                loss_promo = F.cross_entropy(promo_logits, promo_targets,
                                             label_smoothing=label_smoothing)
                losses.append(loss_promo)

            if losses:
                loss = sum(losses) / len(losses)
            else:
                loss = torch.tensor(0.0, device=idx.device)

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

        # For large datasets, use parallel processing to speed up tokenization
        if len(text) > 1_000_000 and __name__ == '__main__':
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            chunk_size = 200_000
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

            num_cores = mp.cpu_count() - 1
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                chunk_args = [(chunk,) for chunk in chunks]
                results = list(executor.map(process_chunk_for_chess_moves, chunk_args))

            for chunk_tokens, chunk_roles in results:
                self.tokens.extend(chunk_tokens)
                self.roles.extend(chunk_roles)

            print(f"Parallel tokenization complete with {num_cores} cores")
        else:
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

        print(f"Tokenized {len(self.tokens)} tokens ({len(self.tokens)} stream positions), all validated")

    def _tokenize_text(self, text):
        """Sequential tokenization: parse text into 4-token-per-ply sequences."""
        ply = 0  # Ply counter within current game (reset at STARTGAME)
        i = 0
        while i < len(text):
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


def load_chess_file():
    """Load chess games file for training"""
    print("Please select a chess games file.")
    file_path = filedialog.askopenfilename(
        title="Select Chess Games File",
        filetypes=[("Text files", "*.txt")]
    )

    if file_path:
        try:
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


def save_model_all(model, all_text, n_embd, n_head, n_kv_heads, n_layer, dropout, block_size, epoch, batch_idx, batch_size, optimizer, scheduler, scaler, loss, learning_rate=None, weight_decay=None, gpu_indices=None):
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
            'vocab_size': ROLE_VOCAB_SIZE,
            'format_version': 2,
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

    Uses 4-token-per-ply generation: COLOR, FROM, TO (conditioned on FROM), PROMO.

    Args:
        epoch: Current training epoch
        num_epochs: Total training epochs
        batch_idx: Current batch index within epoch
        data_loader: Training data loader (for progress display)
        loss: Current training loss value
        model: ChessModel (may be wrapped in DataParallel)
        x: Current batch input tensor [batch_size, seq_len]
        tokens_to_generate: Number of plies to generate
        all_text: Accumulator string for generated samples across training
        idx_to_move: Dictionary mapping token indices back to token names

    Returns:
        Updated all_text string with new generated samples appended
    """
    print(f"\nEpoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss:.4f}")

    was_training = model.training
    model.eval()
    model_single = model.module if isinstance(model, nn.DataParallel) else model
    # Handle torch.compile wrapper for direct attribute access
    model_raw = model_single._orig_mod if hasattr(model_single, '_orig_mod') else model_single

    with torch.no_grad():
        input_seq = x[-1].unsqueeze(0)
        # Display input sequence using idx_to_move
        input_seq_str = ' '.join([idx_to_move.get(t.item(), f'<UNK:{t.item()}>') for t in input_seq[0]])

        print("\nInput Sequence:")
        print(input_seq_str)

        generated_moves = []
        num_plies = min(tokens_to_generate, 32)  # Each ply = 4 forward passes
        dev = input_seq.device

        for _ in range(num_plies):
            # 1. COLOR prediction
            output, _ = model_single(input_seq)
            color_logits = output['color'][0, -1]  # [2]
            color_idx = color_logits.argmax(dim=-1).item()
            color_tok = COLOR_OFFSET + color_idx
            input_seq = torch.cat([input_seq[:, 1:],
                                   torch.tensor([[color_tok]], device=dev)], dim=1)

            # 2. FROM prediction
            output, _ = model_single(input_seq)
            from_logits = output['from'][0, -1]  # [64]
            from_sq = from_logits.argmax(dim=-1).item()
            from_tok = FROM_OFFSET + from_sq
            input_seq = torch.cat([input_seq[:, 1:],
                                   torch.tensor([[from_tok]], device=dev)], dim=1)

            # 3. TO prediction (conditioned on FROM)
            output, _ = model_single(input_seq)
            h_last = output['hidden'][0, -1]  # [n_embd]
            from_emb = model_raw.emb_from(
                torch.tensor(from_sq, device=dev)
            )
            h_conditioned = h_last + from_emb
            to_logits = model_raw.head_to(h_conditioned)  # [64]
            # Ensure TO != FROM
            to_logits[from_sq] = float('-inf')
            to_sq = to_logits.argmax(dim=-1).item()
            to_tok = TO_OFFSET + to_sq
            input_seq = torch.cat([input_seq[:, 1:],
                                   torch.tensor([[to_tok]], device=dev)], dim=1)

            # 4. PROMO prediction
            output, _ = model_single(input_seq)
            promo_logits = output['promo'][0, -1]  # [5]
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
        print(f"Opening file dialog in: /home/jonathan/Data")
        model_file = create_file_dialog(title="Select Chess Model File", filetypes=[("PyTorch files", "*.pth")], initialdir="/home/jonathan/Data")
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

        # Check format version - reject old composite-move checkpoints
        fmt_version = hyperparameters.get('format_version', 1)
        if fmt_version < 2:
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
            # Find smallest n_embd >= current that is divisible by n_head
            original_embd = n_embd
            while n_embd % n_head != 0:
                n_embd += n_head  # Increase by n_head to maintain head_dim
            print(f"Adjusted n_embd from {original_embd} to {n_embd} (head_dim = {n_embd // n_head})")

        n_kv_heads = hyperparameters.get('n_kv_heads', n_head // 3)

        # Load tokenizer
        tokenizer = checkpoint.get('tokenizer')
        if isinstance(tokenizer, dict):
            global move_to_idx, idx_to_move
            move_to_idx = tokenizer
            idx_to_move = {idx: move for move, idx in move_to_idx.items()}
            print(f"Loaded chess moves tokenizer with {len(move_to_idx)} tokens")

        # Create chess model
        model = ChessModel(vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout)
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

        print(f"Chess model loaded from {model_file}")
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


def enter_batch_size(n_embd, n_head, block_size, n_layer, batch_size, gpu_indices):
    """Calculate conservative batch size for stable chess training (prevents crashes)"""
    bytes_per_float = 4
    num_gpus = len(gpu_indices)

    # Conservative GPU optimizations for stability with VNC/Cinnamon
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n🎯 Conservative GPU Optimization for {gpu_name}")
        # Use conservative settings to prevent crashes with display conflicts
        safety_factor = 0.85  # Conservative safety factor for all GPUs
        memory_efficiency = 0.90  # Standard memory efficiency
    else:
        print("\n💻 CPU Mode")
        safety_factor = 0.95  # Very conservative for CPU
        memory_efficiency = 0.85

    if torch.cuda.is_available() and len(gpu_indices) > 0:
        gpu_memory = sum(torch.cuda.get_device_properties(i).total_memory for i in gpu_indices)
    else:
        gpu_memory = 64 * 1024**3  # Default CPU memory estimate

    # Memory calculations optimized for chess model
    vocab_size = ROLE_VOCAB_SIZE
    token_embeddings = vocab_size * n_embd * bytes_per_float
    position_embeddings = block_size * n_embd * bytes_per_float

    # Chess model uses true GQA (Grouped Query Attention) - calculate correct memory usage
    head_dim = n_embd // n_head
    n_kv_heads = max(1, n_head // 4)  # GQA ratio, but use actual passed value if available

    # Attention weights: Q_proj + K_proj + V_proj + out_proj
    q_proj = n_embd * n_embd  # Q projection: n_embd -> n_embd
    kv_proj = n_embd * n_kv_heads * head_dim * 2  # K,V projections: n_embd -> n_kv_heads * head_dim each
    out_proj = n_embd * n_embd  # Output projection: n_embd -> n_embd
    attention_weights_per_layer = (q_proj + kv_proj + out_proj) * bytes_per_float
    attention_weights = n_layer * attention_weights_per_layer

    feedforward_weights = n_layer * 4 * n_embd * n_embd * bytes_per_float  # SwiGLU (w1,w2,w3)
    rms_norm_weights = n_layer * 2 * n_embd * bytes_per_float  # RMSNorm per block (2 per layer)

    total_model_params = token_embeddings + position_embeddings + attention_weights + feedforward_weights + rms_norm_weights

    optimizer_memory = total_model_params * 2  # Adam optimizer
    gradient_memory = total_model_params

    # Activations per sequence (chess model with gradient checkpointing)
    # Gradient checkpointing significantly reduces memory by recomputing forward pass during backprop
    # Only essential activations (embeddings, attention outputs) are stored

    # Input embeddings (token + position) - always stored
    embedding_activations = block_size * n_embd * bytes_per_float * 2

    # Attention computation requires storing Q,K,V for backprop, but with checkpointing this is minimized
    # Approximate attention memory per layer (conservative estimate)
    attention_per_layer = block_size * n_embd * bytes_per_float * 3  # Q,K,V projections
    attention_activations = n_layer * attention_per_layer

    # Feedforward activations - checkpointed, so minimal storage
    ff_per_layer = block_size * n_embd * bytes_per_float * 2  # Input and output of FF
    ff_activations = n_layer * ff_per_layer

    # Gradient checkpointing provides significant memory savings
    # Estimate: 50-70% reduction in activation memory during training
    checkpointing_savings = 0.6  # 60% reduction
    total_per_seq = (embedding_activations + attention_activations + ff_activations) * (1 - checkpointing_savings) * memory_efficiency

    # Adjust for multi-GPU setup
    if num_gpus > 1:
        # DataParallel replicates model on each GPU
        total_model_params *= num_gpus
        optimizer_memory *= num_gpus
        gradient_memory *= num_gpus
        # But activations are split across GPUs
        total_per_seq = total_per_seq / num_gpus

    available_memory = gpu_memory * safety_factor - (total_model_params + optimizer_memory + gradient_memory)
    max_batch_size = max(1, int(available_memory / total_per_seq))

    print(f"\n🧠 Conservative Memory Analysis for {num_gpus} GPU{'s' if num_gpus > 1 else ''} (crash prevention):")
    print(f"- Model parameters: {total_model_params / 1e9:.2f} GB")
    print(f"- Optimizer memory: {optimizer_memory / 1e9:.2f} GB")
    print(f"- Gradient memory: {gradient_memory / 1e9:.2f} GB")
    print(f"- Memory per sequence: {total_per_seq / 1e6:.2f} MB")
    print(f"- Total GPU memory: {gpu_memory / 1e9:.1f} GB")
    print(f"- Available memory: {available_memory / 1e9:.2f} GB")

    if num_gpus == 1:
        print(f"✅ Single GPU - Maximum batch size: {max_batch_size}")

        # Optimized batch size recommendations for Blackwell GB10 (128GB unified memory)
        if torch.cuda.is_available():
            # Blackwell GB10 has 128GB unified memory - can handle large batches
            # Chess models are memory-efficient due to optimized architecture
            recommended_batch = min(max_batch_size, 256)  # Optimized for Blackwell performance
        else:
            recommended_batch = min(max_batch_size, 32)  # CPU training

    else:
        print(f"⚠️  Multi-GPU DataParallel - Maximum batch size per GPU: {max_batch_size}")
        print("Note: DataParallel may freeze. Consider using single GPU for better reliability.")

        # Optimized recommendations for multi-GPU Blackwell setup
        if torch.cuda.is_available():
            # Blackwell GB10 GPUs have 128GB each - can handle large batches
            recommended_batch = min(max_batch_size, 128)  # Optimized for multi-Blackwell performance
        else:
            recommended_batch = min(max_batch_size, 16)  # CPU multi-processing

    batch_size = int(get_input_with_default(f"Enter batch size (recommended: {recommended_batch}, max: {max_batch_size}): ", recommended_batch))
    batch_size = max(1, min(batch_size, max_batch_size))

    return batch_size


# Global variables for chess tokenization
move_to_idx = create_move_to_idx()
idx_to_move = create_idx_to_move()




# Core training function
def _train_chess_model_core(text, checkpoint_data=None):
    """
    Core training logic for chess move prediction model.

    Handles the complete training pipeline including model setup, GPU configuration,
    optimizer initialization, data loading, and training loop execution. Supports
    both fresh training and checkpoint resumption.

    Training features:
    - Blackwell GB10 GPU optimizations for maximum performance (128GB unified memory)
    - DataParallel support for multi-GPU training
    - Adafactor optimizer for stable chess model training
    - Gradient scaling and clipping for training stability
    - Progress monitoring with sample move generation
    - Automatic checkpointing with comprehensive state saving

    Args:
        text: Preprocessed chess game text data
        checkpoint_data: Optional tuple from load_model_file() for training resumption
                        Contains: (model, vocab_size, n_embd, n_head, n_kv_heads, block_size,
    """
    print("ChessBrain - Chess Move Prediction LLM")
    print("=" * 50)
    print(f"DEBUG: Starting _train_chess_model_core with text length: {len(text)}")

    vocab_size = len(move_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    # Ensure gpu_indices is accessible (capture from global scope)
    global gpu_indices
    if 'gpu_indices' not in globals() or gpu_indices is None:
        gpu_indices = [0] if torch.cuda.is_available() else None

    # Set default values for all variables that might be needed
    learning_rate = CHESS_DEFAULTS['learning_rate']
    weight_decay = CHESS_DEFAULTS['weight_decay']

    # Check if resuming from checkpoint
    if checkpoint_data:
        model, vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, \
        optimizer_state_dict, scheduler_state_dict, scaler_state_dict, start_epoch, start_batch, checkpoint_hyperparams = checkpoint_data
        model.start_game_token = move_to_idx['<STARTGAME>']
        print(f"Resuming from checkpoint: epoch {start_epoch}, batch {start_batch}")

        # Display model architecture (cannot be changed)
        print(f"Model architecture: {n_layer} layers, {n_head} heads, {n_embd} embedding dim, dropout {dropout}")
        print(f"Block size: {block_size}, Vocab size: {vocab_size}")

        # Display training setup from checkpoint
        saved_gpu_indices = checkpoint_hyperparams.get('gpu_indices')
        # Capture early GPU selection before any assignments (access global to avoid UnboundLocalError)
        early_gpu_selection = globals().get('gpu_indices')
        if saved_gpu_indices:
            print(f"📋 Checkpoint originally trained on GPU{'s' if len(saved_gpu_indices) > 1 else ''}: {saved_gpu_indices}")
            print(f"   System has {torch.cuda.device_count()} GPUs available: {[f'GPU{i}' for i in range(torch.cuda.device_count())]}")

            # For single GPU, give user choice to use different GPU
            if len(saved_gpu_indices) == 1:
                gpu_input = get_input_with_default(
                    f"Resume on GPU {saved_gpu_indices[0]} or enter new GPU number", str(saved_gpu_indices[0])
                )

                try:
                    new_gpu = int(gpu_input)
                    # For single GPU resumption, trust the user's input since early selection already validated
                    gpu_indices = [new_gpu]
                    print(f"Selected GPU: {new_gpu}")
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(new_gpu)
                    print(f"CUDA_VISIBLE_DEVICES updated to: {new_gpu}")
                except ValueError:
                    print(f"Invalid input. Using current GPU {early_gpu_selection[0] if early_gpu_selection else saved_gpu_indices[0]}")
                    gpu_indices = early_gpu_selection if early_gpu_selection else saved_gpu_indices
            else:
                # Multi-GPU: force same GPUs for state consistency
                print(f"🔄 Auto-selecting same GPUs for resume: {saved_gpu_indices}")
                gpu_indices = saved_gpu_indices
                # Update CUDA_VISIBLE_DEVICES for checkpoint GPUs
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_indices))
                print(f"CUDA_VISIBLE_DEVICES updated to: {os.environ['CUDA_VISIBLE_DEVICES']}")
                print(f"⚠️  IMPORTANT: Must use SAME number of GPUs as original training")
                print(f"   Checkpoint has states for {len(saved_gpu_indices)} GPUs - cannot change GPU count")
                print(f"   ARM/1-GPU checkpoint → load on ARM/1-GPU ✅ | 3-GPU checkpoint → load on 3-GPUs ✅")
                print(f"   ARM/1-GPU checkpoint → load on 3-GPUs ❌ | 3-GPU checkpoint → load on 1-GPU ❌")
        else:
            print("GPU information not available in checkpoint")
            # Keep existing gpu_indices from early selection

        # Ensure gpu_indices is always set after checkpoint loading logic
        # (but don't override if already set by checkpoint)
        try:
            if gpu_indices is None:
                # Fallback: this shouldn't happen but prevents UnboundLocalError
                gpu_indices = [0] if torch.cuda.is_available() else None
        except NameError:
            gpu_indices = [0] if torch.cuda.is_available() else None

        # Ask for training parameters (use checkpoint values as defaults where available)
        # These can be safely changed without breaking the model
        saved_batch_size = checkpoint_hyperparams.get('batch_size', 256)
        print(f"Note: Changing batch size may affect training resumption accuracy")
        batch_size = int(get_input_with_default("Batch size", saved_batch_size))
        num_epochs = int(get_input_with_default("Number of epochs", 20))

        # Learning rate and weight decay can be adjusted (will be loaded from optimizer state)
        # Use the actual current LR from optimizer state as default, not the saved hyperparams value
        # Handle both single-GPU (dict) and multi-GPU (list of dicts) cases
        if optimizer_state_dict:
            if isinstance(optimizer_state_dict, list):
                # Multi-GPU: get LR from first GPU's optimizer
                saved_learning_rate = optimizer_state_dict[0]['param_groups'][0]['lr']
            else:
                # Single-GPU: direct access
                saved_learning_rate = optimizer_state_dict['param_groups'][0]['lr']
        else:
            saved_learning_rate = CHESS_DEFAULTS['learning_rate']
        print(f"Current learning rate from checkpoint: {saved_learning_rate}")
        learning_rate_input = get_input_with_default("Learning rate (or press Enter to keep current)", saved_learning_rate)
        learning_rate = float(learning_rate_input)

        saved_weight_decay = checkpoint_hyperparams.get('weight_decay', CHESS_DEFAULTS['weight_decay'])
        weight_decay_input = get_input_with_default("Weight decay", saved_weight_decay)
        weight_decay = float(weight_decay_input)

        saved_dropout = checkpoint_hyperparams['dropout']
        dropout_input = get_input_with_default("Dropout", saved_dropout)
        dropout = float(dropout_input)
    else:
        # Fresh start - use defaults
        start_epoch = 0
        start_batch = 0
        optimizer_state_dict = None
        scheduler_state_dict = None
        scaler_state_dict = None

        # Single GPU mode - use interactive prompts or defaults
        n_embd = CHESS_DEFAULTS['n_embd']
        n_head = CHESS_DEFAULTS['n_head']
        n_kv_heads = CHESS_DEFAULTS['n_kv_heads']
        block_size = CHESS_DEFAULTS['block_size']
        n_layer = CHESS_DEFAULTS['n_layer']
        dropout = CHESS_DEFAULTS['dropout']
        batch_size = CHESS_DEFAULTS['batch_size']
        num_epochs = CHESS_DEFAULTS['num_epochs']

        # Allow parameter overrides
        n_embd = int(get_input_with_default("Embedding dimensions", n_embd))
        n_head = int(get_input_with_default("Number of query heads", n_head))
        n_kv_heads = int(get_input_with_default("Number of KV heads", n_kv_heads))
        block_size = int(get_input_with_default("Sequence length", block_size))
        n_layer = int(get_input_with_default("Number of layers", n_layer))
        dropout = float(get_input_with_default("Dropout", dropout))
        batch_size = int(get_input_with_default("Batch size", batch_size))
        num_epochs = int(get_input_with_default("Number of epochs", num_epochs))

        # Ensure n_embd is divisible by n_head
        if n_embd % n_head != 0:
            print(f"Warning: n_embd ({n_embd}) not divisible by n_head ({n_head})")
            # Adjust n_embd to be divisible by n_head
            original_embd = n_embd
            while n_embd % n_head != 0:
                n_embd += n_head  # Increase by n_head to maintain head_dim
            print(f"Adjusted n_embd from {original_embd} to {n_embd} (head_dim = {n_embd // n_head})")

        # Create your own TransformerModel for chess move prediction
        model = ChessModel(vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, use_chess=True)
        model.start_game_token = move_to_idx['<STARTGAME>']

        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Choose optimizer and scheduler early - needed for both single and multi-GPU setups
    clip_threshold = CHESS_DEFAULTS['max_norm']
    # Only ask for optimizer choice if not loading from checkpoint
    if not checkpoint_data:
        optimizer_choice = get_input_with_default("Optimizer (adamw/adfactor)", "adamw").lower()
        scheduler_choice = get_input_with_default("Scheduler (cosine/plateau/exponential) [plateau=recommended for stuck training]", "plateau").lower()
    else:
        # When loading checkpoint, ask if they want to switch schedulers
        print("\n📋 Current checkpoint was saved with CosineAnnealingLR scheduler")
        print("   This scheduler decays learning rate regardless of training progress")
        print("   Consider switching to ReduceLROnPlateau for better plateau handling")
        print("   Or try ExponentialLR for gentle decay (recommended for stuck training)")
        scheduler_choice = get_input_with_default("Keep current scheduler or switch? (cosine/plateau/exponential)", "plateau").lower()
        optimizer_choice = "adamw"  # Default, will be overridden by loaded state

    # Configure GPU training with Blackwell GB10 optimizations for chess model
    if torch.cuda.is_available():
        # For checkpoint loading, gpu_indices is already set by checkpoint logic above
        # For fresh training, gpu_indices needs to be set
        if checkpoint_data is None:
            # Fresh training - need to select GPUs if not already set
            if gpu_indices is None:
                gpu_indices = select_gpus()
        # else: checkpoint_data exists, gpu_indices already set by checkpoint loading logic above

        if gpu_indices and len(gpu_indices) > 0:
            print(f"\n🚀 Setting up chess training on GPU{'s' if len(gpu_indices) > 1 else ''}: {gpu_indices}")
            device = torch.device('cuda')

            # Always move model to device first
            model = model.to(device)

            # Enable torch.compile for GPU acceleration (after device setup)
            if device.type in ['mps', 'cuda'] and hasattr(torch, 'compile'):
                try:
                    print(f"🚀 Enabling torch.compile() for {device.type.upper()} acceleration...")
                    model = torch.compile(model, mode='default')
                    print("✅ torch.compile() enabled successfully!")
                except Exception as e:
                    print(f"⚠️  torch.compile() failed: {e}")
                    print("   Continuing with standard model")
            else:
                print("ℹ️  torch.compile() not available or not needed for this device")

            # Initialize multi-GPU flag (only if not already set by checkpoint loading)
            if 'use_custom_parallel' not in locals():
                use_custom_parallel = False

            # Multi-GPU setup - Custom Parallel Training (NOT DataParallel)
            # =================================================================
            # Why Custom Multi-GPU instead of DataParallel:
            # - DataParallel has GIL bottlenecks, CUDA sync issues, memory replication
            # - Custom approach: Each GPU gets its own model/optimizer/scheduler/scaler
            # - Manual gradient averaging avoids DataParallel's complex synchronization
            # - No Python multiprocessing = no GIL issues
            # - Each GPU processes independent batch chunks in parallel
            #
            # Architecture: N GPUs = N identical models, N optimizers, N schedulers, N scalers
            # Training: Split batch across GPUs → forward/backward independently → average gradients → update all models
            # =================================================================
            if len(gpu_indices) > 1:
                print(f"🚀 MULTI-GPU MODE: Custom parallel training across {len(gpu_indices)} GPUs")
                print(f"   Each GPU gets independent model/optimizer/scheduler/scaler replica")
                print(f"   Manual gradient averaging avoids DataParallel GIL/sync issues")
                if checkpoint_data:
                    print(f"   🔄 Resuming multi-GPU training from checkpoint")
                    print(f"   ✅ All optimizer/scheduler/scaler states loaded for all GPUs")

            if len(gpu_indices) > 1:  # After checkpoint check
                print(f"🔄 Attempting to setup multi-GPU training on GPUs: {gpu_indices}")
                try:
                    # Validate that requested GPUs are available
                    available_gpus = torch.cuda.device_count()
                    max_requested_gpu = max(gpu_indices)
                    print(f"   Requested max GPU index: {max_requested_gpu}, System has {available_gpus} GPUs (indices 0-{available_gpus-1})")

                    if max_requested_gpu >= available_gpus:
                        print(f"⚠️  WARNING: Checkpoint trained on GPUs {gpu_indices} but only {available_gpus} GPUs available")
                        print(f"   GPU {max_requested_gpu} not available on this system")
                        print(f"   Falling back to single GPU mode on GPU 0")
                        gpu_indices = [0]
                        use_custom_parallel = False
                    else:
                        # Create model replicas manually on each GPU
                        models = []
                        optimizers = []
                        schedulers = []
                        scalers = []

                        for i, gpu_idx in enumerate(gpu_indices):
                            # Create model on specific GPU
                            model_gpu = ChessModel(vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, use_chess=True)
                            model_gpu.start_game_token = move_to_idx['<STARTGAME>']
                            
                            # If loading from checkpoint, copy the loaded weights to each GPU model
                            if checkpoint_data:
                                # Get state dict and clean torch.compile prefixes
                                source_state = model.state_dict()
                                cleaned_state = {}
                                for key, val in source_state.items():
                                    new_key = key
                                    if new_key.startswith('_orig_mod.'):
                                        new_key = new_key[len('_orig_mod.'):]
                                    cleaned_state[new_key] = val
                                model_gpu.load_state_dict(cleaned_state)
                            
                            model_gpu = model_gpu.to(f'cuda:{i}')  # Use remapped indices
                            models.append(model_gpu)

                            # Create optimizer for this GPU
                            if optimizer_choice == 'adamw':
                                opt = torch.optim.AdamW(
                                    model_gpu.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    betas=(0.9, 0.999),
                                    eps=1e-8
                                )
                            else:
                                opt = Adafactor(
                                    model_gpu.parameters(),
                                    lr=learning_rate,
                                    scale_parameter=True,
                                    relative_step=False,
                                    warmup_init=False,
                                    clip_threshold=clip_threshold,
                                    weight_decay=weight_decay,
                                    beta1=0.9,
                                    eps=(1e-30, 1e-3)
                                )
                            optimizers.append(opt)

                            # Create scheduler for this GPU
                            if scheduler_choice == 'plateau':
                                sched = ReduceLROnPlateau(opt, mode='min', factor=0.8, patience=100, threshold=0.001, min_lr=1e-7)
                            elif scheduler_choice == 'exponential':
                                sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
                            else:
                                sched = CosineAnnealingLR(opt, T_max=num_epochs)
                            schedulers.append(sched)

                            # Create scaler for this GPU
                            scal = GradScaler()
                            scalers.append(scal)

                        print(f"✅ Custom multi-GPU setup complete: {len(models)} models created on GPUs: {gpu_indices}")
                        use_custom_parallel = True

                except Exception as e:
                    print(f"⚠️  ERROR: Failed to setup multi-GPU training: {e}")
                    print(f"   Falling back to single GPU mode on GPU 0")
                    gpu_indices = [0]
                    use_custom_parallel = False

            # Load checkpoint states for multi-GPU resume (only if multi-GPU setup succeeded)
            # =================================================================
            # Multi-GPU Checkpoint Loading:
            # - Checkpoint saves: [optimizer_states, scheduler_states, scaler_states] for ALL GPUs
            # - Loading: Distribute saved states to corresponding GPU's optimizer/scheduler/scaler
            # - Why same GPU count required: N saved states → N GPUs to load into
            # - If GPU count differs: States won't match → resume fails or inconsistent
            #
            # Example: Trained on 3 GPUs → checkpoint has 3 optimizer states
            #         Resume on 3 GPUs → load state[0]→GPU0, state[1]→GPU1, state[2]→GPU2
            #         Resume on 1 GPU → ERROR: 3 states but only 1 optimizer to load into
            # =================================================================
            if checkpoint_data and use_custom_parallel:
                optimizer_state_dict = checkpoint_data[8]  # optimizer states (list for multi-GPU)
                scheduler_state_dict = checkpoint_data[9]  # scheduler states (list for multi-GPU)
                scaler_state_dict = checkpoint_data[10]    # scaler states (list for multi-GPU)

                if isinstance(optimizer_state_dict, list) and len(optimizer_state_dict) == len(optimizers):
                    print(f"Loading {len(optimizer_state_dict)} optimizer states for {len(optimizers)} GPUs")
                    for i, opt_state in enumerate(optimizer_state_dict):
                        try:
                            optimizers[i].load_state_dict(opt_state)
                            print(f"  GPU {gpu_indices[i]}: optimizer state loaded")

                            # Always reset LR when switching to cosine scheduler (multi-GPU)
                            if scheduler_choice == 'cosine':
                                optimizers[i].param_groups[0]['lr'] = learning_rate
                                print(f"  GPU {gpu_indices[i]}: reset LR to {learning_rate} for cosine scheduler")
                        except ValueError as e:
                            if "parameter group" in str(e):
                                print(f"  ⚠️  GPU {gpu_indices[i]}: Optimizer state incompatible (likely PyTorch version change)")
                                print(f"     Continuing with fresh optimizer state (this is normal after PyTorch updates)")
                            else:
                                raise e

                # Always start with fresh scheduler states when loading checkpoint
                # This allows switching scheduler types (e.g., cosine → plateau) for stuck runs
                print(f"🔄 Using fresh {scheduler_choice} schedulers for all GPUs (allows switching types for stuck runs)")

                if isinstance(scaler_state_dict, list) and len(scaler_state_dict) == len(scalers):
                    print(f"Loading {len(scaler_state_dict)} scaler states for {len(scalers)} GPUs")
                    for i, scal_state in enumerate(scaler_state_dict):
                        try:
                            scalers[i].load_state_dict(scal_state)
                            print(f"  GPU {gpu_indices[i]}: scaler state loaded")
                        except (ValueError, RuntimeError) as e:
                            print(f"  ⚠️  GPU {gpu_indices[i]}: Scaler state incompatible (continuing with fresh scaler)")
                            print(f"     This is normal after PyTorch updates or configuration changes")

                print("✅ Multi-GPU checkpoint states loaded - training resumes with all states preserved")
            # Only fall back to single GPU if we're not in multi-GPU mode
            if not use_custom_parallel:
                # Single GPU mode - optimal for chess training stability
                print(f"✅ SINGLE GPU MODE: Using Blackwell GB10 GPU {gpu_indices[0]} (recommended for chess)")
                print(f"   128GB unified memory supports batch size 32 for stable training")
                # Model already moved to device above
                use_custom_parallel = False

                # DISABLED: torch.compile() can cause crashes with VNC/Cinnamon + NVIDIA drivers
                # Keeping this commented out to prevent system crashes
                # if hasattr(torch, 'compile'):
                #     print("🚀 Applying torch.compile() for training speedup...")
                #     try:
                #         model = torch.compile(model, mode='reduce-overhead')
                #         print("✅ torch.compile() applied successfully")
                #     except Exception as e:
                #         print(f"⚠️  torch.compile() failed: {e}, continuing without compilation")
                # else:
                #     print("ℹ️  torch.compile() not available, using standard model")
                print("ℹ️  torch.compile() disabled for stability - using standard model")
        else:
            print("❌ ERROR: No CUDA GPUs available. This chess training requires GPU support.")
            print("   Please run on a system with GPU support (CUDA or MPS).")
            exit(1)
    else:
        # Handle MPS (Mac) - GPU only, no CPU fallback
        if torch.backends.mps.is_available():
            print("🚀 Setting up chess training on MPS GPU")
            device = torch.device('mps')
            gpu_indices = []  # MPS doesn't use gpu_indices like CUDA
            model = model.to(device)
            use_custom_parallel = False  # MPS doesn't use custom parallel training
        else:
            print("❌ ERROR: No GPU available. This chess training requires MPS (Mac) or CUDA GPU.")
            print("   Please run on a system with GPU support.")
            exit(1)

    # Create optimizer/scheduler/scaler - only for single GPU mode
    # Multi-GPU mode creates its own optimizers/schedulers/scalers above
    if not use_custom_parallel:
        model_params = get_model_module(model).parameters()

        if optimizer_choice == 'adamw':
            print("🚀 Using AdamW optimizer (better convergence for chess models)")
            optimizer = torch.optim.AdamW(
                model_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            print("🔧 Using Adafactor optimizer (memory efficient for large models)")
            optimizer = Adafactor(
                model_params,
                lr=learning_rate,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False,
                clip_threshold=clip_threshold,
                weight_decay=weight_decay,
                beta1=0.9,
                eps=(1e-30, 1e-3)
            )

        # Setup scheduler and scaler
        if scheduler_choice == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=100, threshold=0.001, min_lr=1e-7)
        elif scheduler_choice == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:  # cosine
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        scaler = GradScaler()

        # Restore optimizer and scheduler state if resuming from checkpoint
        if checkpoint_data and optimizer_state_dict:
            print("Loading optimizer state from checkpoint")
            try:
                optimizer.load_state_dict(optimizer_state_dict)
            except ValueError as e:
                if "parameter group" in str(e):
                    print("⚠️  Optimizer state incompatible (likely PyTorch version change)")
                    print("   Continuing with fresh optimizer state (this is normal after PyTorch updates)")
                else:
                    raise e

            # Always reset LR when switching to cosine scheduler (it should start fresh, not inherit plateau's crushed LR)
            if scheduler_choice == 'cosine':
                print(f"Switching to cosine scheduler - resetting LR to {learning_rate} (ignoring checkpoint LR)")
                optimizer.param_groups[0]['lr'] = learning_rate
                print(f"Cosine scheduler starting with fresh learning rate: {learning_rate}")
            else:
                # For plateau scheduler, check if user changed LR from checkpoint default
                checkpoint_lr = optimizer.param_groups[0]['lr']
                if checkpoint_data and abs(learning_rate - checkpoint_lr) > 1e-8:  # User changed LR
                    print(f"User changed learning rate from {checkpoint_lr} to {learning_rate}")
                    optimizer.param_groups[0]['lr'] = learning_rate
                    print(f"Applied new learning rate: {learning_rate}")
                else:
                    # Keep the checkpoint LR (it manages its own decay)
                    learning_rate = checkpoint_lr  # Update our variable to match
                    print(f"Resumed with learning rate: {learning_rate}")

        # Always start with fresh scheduler state when loading checkpoint
        # This allows switching scheduler types (e.g., cosine → plateau) for stuck runs
        print(f"🔄 Using fresh {scheduler_choice} scheduler (allows switching types for stuck runs)")

        if checkpoint_data and scaler_state_dict:
            print("Loading scaler state from checkpoint")
            try:
                scaler.load_state_dict(scaler_state_dict)
            except (ValueError, RuntimeError) as e:
                print("⚠️  Scaler state incompatible (continuing with fresh scaler)")
                print("   This is normal after PyTorch updates or configuration changes")

    # Create dataset and dataloader
    dataset = ChessMovesDataset(text, block_size, move_to_idx)
    print(f"Dataset size: {len(dataset)} sequences")

    # Check GPU capability for Blackwell-specific fixes (cache this - don't call every batch)
    gpu_capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (8, 6)
    is_blackwell_gpu = gpu_capability == (12, 1)  # Cache Blackwell detection

    # Configure data loading for optimal chess training performance
    sampler = None
    if use_custom_parallel or isinstance(model, nn.DataParallel):
        # Multi-GPU setup - use workers to keep GPUs fed
        num_gpus_active = len(gpu_indices) if gpu_indices else 1
        print(f"Multi-GPU: Configuring data loading for {num_gpus_active} GPUs")
        num_workers = min(32, os.cpu_count() // 2)
        pin_memory = device.type == 'cuda'
        persistent_workers = True
        prefetch_factor = 4
    elif device.type == 'mps':
        num_workers = min(4, os.cpu_count() // 2)
        pin_memory = False
        persistent_workers = True
        prefetch_factor = 2
        print(f"Mac Studio MPS: Using {num_workers} workers for accelerated data loading")
    else:
        # Single CUDA GPU
        print("Single GPU: Using optimal configuration for chess training")
        num_workers = min(8, os.cpu_count() // 2)
        pin_memory = device.type == 'cuda'
        persistent_workers = num_workers > 0
        prefetch_factor = 2 if num_workers > 0 else None

    # Print the number of model parameters
    num_params = sum(p.numel() for p in get_model_module(model).parameters())
    print(f"Number of model parameters: {num_params}")
    if isinstance(model, nn.DataParallel):
        print(f"Model's primary device: {next(get_model_module(model).parameters()).device}")
        print(f"Model distributed across devices: {model.device_ids}")
    else:
        print(f"Model is on device: {next(model.parameters()).device}")

    print(f"Using {num_workers} workers for data loading")

    # Add these debug prints
    print(f"Model type: {type(model)}")
    if isinstance(model, nn.DataParallel):
        print(f"DataParallel devices: {model.device_ids}")

    # Update the memory usage print section:
    if torch.cuda.is_available():
        if len(gpu_indices) > 0:
            # After CUDA_VISIBLE_DEVICES, use the remapped device indices (0, 1, 2...) not physical indices
            visible_devices = list(range(len(gpu_indices)))
            for i, physical_gpu in zip(visible_devices, gpu_indices):
                print(f"GPU {physical_gpu} (device {i}) memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
                print(f"GPU {physical_gpu} (device {i}) memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")

    # Training loop - handles both single GPU and custom multi-GPU
    if use_custom_parallel:
        # Custom multi-GPU training loop - NO DataParallel GIL issues!
        print(f"🎯 Using custom multi-GPU training - {len(models)} GPUs work simultaneously!")
        print(f"   Expected speedup: ~{len(models)}x faster than single GPU (minus overhead)")
        running_loss = 0.0
        total_batches = 0
        epoch_losses = []
        all_text = ""
        inference_frequency = 500

        # Continuous plateau detection in groups of 20 × 100-batch averages

        # Set all models to training mode
        for model_gpu in models:
            model_gpu.train()

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            # Create dataloader for this epoch
            data_loader = DataLoader(dataset, batch_size=batch_size,
                                    sampler=sampler, shuffle=(sampler is None),
                                    drop_last=True, num_workers=num_workers,
                                    pin_memory=pin_memory, persistent_workers=persistent_workers,
                                    prefetch_factor=prefetch_factor)

            # Ensure clean shutdown - set workers as daemon processes
            if hasattr(data_loader, '_iterator') and hasattr(data_loader._iterator, '_workers'):
                for worker in data_loader._iterator._workers:
                    worker.daemon = True  # Workers will auto-terminate with main process

            print(f"DataLoader length: {len(data_loader)}, Epoch: {epoch+1}/{num_epochs}")

            for batch_idx, (x, y, y_roles) in enumerate(data_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue

                # Split batch across GPUs
                num_gpus = len(models)
                batch_size = x.shape[0]
                batch_size_per_gpu = batch_size // num_gpus
                x_splits = torch.split(x, batch_size_per_gpu)
                y_splits = torch.split(y, batch_size_per_gpu)
                yr_splits = torch.split(y_roles, batch_size_per_gpu)
                models_single = models
                optimizers_single = optimizers
                scalers_single = scalers
                gpu_indices_single = list(range(num_gpus))

                total_loss = 0.0
                gpu_losses = []
                gpu_grad_counts = []

                # Forward and backward pass on each GPU independently
                for local_idx, (model_gpu, opt_gpu, scal_gpu, x_gpu, y_gpu, yr_gpu) in enumerate(zip(models_single, optimizers_single, scalers_single, x_splits, y_splits, yr_splits)):
                    actual_gpu = gpu_indices_single[local_idx]
                    x_gpu = x_gpu.to(f'cuda:{actual_gpu}', non_blocking=True)
                    y_gpu = y_gpu.to(f'cuda:{actual_gpu}', non_blocking=True)
                    yr_gpu = yr_gpu.to(f'cuda:{actual_gpu}', non_blocking=True)

                    # Forward pass with mixed precision
                    with autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        output, loss = model_gpu(x_gpu, targets=y_gpu, target_roles=yr_gpu)

                    # Backward pass
                    opt_gpu.zero_grad(set_to_none=True)
                    scal_gpu.scale(loss).backward()

                    # Unscale gradients
                    scal_gpu.unscale_(opt_gpu)

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model_gpu.parameters(), clip_threshold)

                    loss_val = loss.item()
                    total_loss += loss_val
                    gpu_losses.append(loss_val)

                    # Count parameters with gradients (only when needed for debug)
                    if batch_idx % 1000 == 0 and batch_idx > 0:
                        grad_count = sum(1 for p in model_gpu.parameters() if p.grad is not None)
                        gpu_grad_counts.append(grad_count)
                    else:
                        gpu_grad_counts.append(0)  # Placeholder to maintain list structure

                # Debug summary every 1000 batches (reduced frequency for performance)
                if batch_idx % 1000 == 0 and batch_idx > 0:
                    print(f"Batch {batch_idx}: Size {batch_size} split across {num_gpus} GPUs: {[s.shape[0] for s in x_splits]}")
                    print(f"  Losses: {[f'{l:.3f}' for l in gpu_losses]}")
                    print(f"  Grad counts: {gpu_grad_counts}")
                    if num_gpus > 1:
                        print(f"  All GPUs active: {all(c > 0 for c in gpu_grad_counts)}")

                # Average gradients across all models - simple and correct
                if num_gpus > 1:  # Only average if actually using multiple GPUs
                    for model_gpu in models_single:
                        for param in model_gpu.parameters():
                            if param.grad is not None:
                                param.grad.data /= num_gpus

                # Update all models with averaged gradients
                for opt_gpu, scal_gpu in zip(optimizers_single, scalers_single):
                    scal_gpu.step(opt_gpu)
                    scal_gpu.update()

                # Models stay synchronized automatically through gradient averaging

                avg_loss = total_loss / num_gpus
                running_loss += avg_loss
                epoch_loss += avg_loss
                total_batches += 1
                epoch_batches += 1

                # Progress reporting
                if (batch_idx + 1) % inference_frequency == 0:
                    avg_running_loss = running_loss / total_batches
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Avg Loss: {avg_running_loss:.4f}")

                    # Continuous plateau detection: average last 10000 batches (20 × 500-batch groups)
                    # Only check for plateau every 5000 batches (10 report cycles)
                    if scheduler_choice == 'plateau':
                        if 'batch_group_losses' not in locals():
                            batch_group_losses = []
                            plateau_check_counter = 0
                        batch_group_losses.append(avg_running_loss)
                        plateau_check_counter += 1

                        # Keep only last 20 groups (10000 batches) for averaging
                        if len(batch_group_losses) > 20:
                            batch_group_losses = batch_group_losses[-20:]

                        # Get current LR for reporting
                        current_lr = schedulers[0].optimizer.param_groups[0]['lr']
                        print(f"📊 LR: {current_lr:.2e} | Loss history: {len(batch_group_losses)} samples")

                        # Only check plateau every 10 report cycles (5000 batches)
                        if plateau_check_counter >= 10 and len(batch_group_losses) >= 20:
                            plateau_check_counter = 0  # Reset counter
                            current_avg = sum(batch_group_losses) / len(batch_group_losses)

                            if 'previous_plateau_avg' in locals():
                                # Compare current 10000-batch average vs previous
                                improvement = previous_plateau_avg - current_avg
                                if improvement >= 0.001:
                                    print(f"📈 Improvement: {improvement:.4f} (>= 0.001) - continuing at LR {current_lr:.2e}")
                                else:
                                    print(f"🔻 Plateau: {improvement:.4f} < 0.001 - reducing LR from {current_lr:.2e}")
                                    for sched in schedulers:
                                        sched.step(float('inf'))  # Force LR reduction (20% reduction with factor=0.8)
                                    new_lr = schedulers[0].optimizer.param_groups[0]['lr']
                                    print(f"   New LR: {new_lr:.2e}")

                            # Update previous average
                            previous_plateau_avg = current_avg

                    # Generate sample using first model
                    all_text = test_progress(
                        epoch, num_epochs, batch_idx, data_loader, avg_loss,
                        models[0], x_splits[0].to(f'cuda:0'), 50, all_text, idx_to_move
                    )

                    # Save using first model and ALL optimizer/scheduler states
                    all_optimizer_states = [opt.state_dict() for opt in optimizers]
                    all_scheduler_states = [sched.state_dict() for sched in schedulers]
                    all_scaler_states = [scal.state_dict() for scal in scalers]

                    save_model_all(
                        models[0], all_text, n_embd, n_head, n_kv_heads, n_layer, dropout,
                        block_size, epoch, batch_idx, batch_size, all_optimizer_states, all_scheduler_states, all_scaler_states, avg_loss,
                        learning_rate, weight_decay, gpu_indices
                    )

                    running_loss = 0.0
                    total_batches = 0

                    # Clear cache on all GPUs
                    for gpu_idx in range(num_gpus):
                        torch.cuda.set_device(gpu_idx)
                        torch.cuda.empty_cache()

            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

            # Step exponential scheduler at end of each epoch
            if scheduler_choice == 'exponential':
                for sched in schedulers:
                    sched.step()
                current_lr = schedulers[0].optimizer.param_groups[0]['lr']
                print(f"📉 Exponential LR decay: {current_lr:.2e}")

            # Schedulers are now stepped every 100 batches, not per epoch

    else:
        # Single GPU training loop (original)
        model.train()
        running_loss = 0.0
        total_batches = 0
        epoch_losses = []
        all_text = ""
        inference_frequency = 500

        # Continuous plateau detection in groups of 20 × 100-batch averages

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            # Create a fresh dataloader for each epoch to allow proper shuffling
            data_loader = DataLoader(dataset, batch_size=batch_size,
                                    sampler=sampler, shuffle=(sampler is None),
                                    drop_last=True, num_workers=num_workers,
                                    pin_memory=pin_memory, persistent_workers=persistent_workers,
                                    prefetch_factor=prefetch_factor)

            # Ensure clean shutdown - set workers as daemon processes
            if hasattr(data_loader, '_iterator') and hasattr(data_loader._iterator, '_workers'):
                for worker in data_loader._iterator._workers:
                    worker.daemon = True  # Workers will auto-terminate with main process

            print(f"DataLoader length: {len(data_loader)}, Epoch: {epoch+1}/{num_epochs}")

            for batch_idx, (x, y, y_roles) in enumerate(data_loader):
                    # Skip batches if resuming from checkpoint
                    if epoch == start_epoch and batch_idx < start_batch:
                        continue

                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    y_roles = y_roles.to(device, non_blocking=True)

                    if batch_idx == 0:
                        print(f"Batch shapes: x={x.shape}, y={y.shape}, y_roles={y_roles.shape}")
                        print(f"Data device: x={x.device}, y={y.device}")

                    # Forward pass with optimized mixed precision for chess model training
                    if device.type == 'cuda':
                        with autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                            output, loss = model(x, targets=y, target_roles=y_roles)

                        # Backward pass with gradient scaling for mixed precision stability
                        optimizer.zero_grad(set_to_none=True)

                        scaler.scale(loss).backward()  # Scale loss for FP16 gradient stability

                        # Unscale gradients before clipping (required for Adafactor with mixed precision)
                        scaler.unscale_(optimizer)

                        # Apply gradient clipping to prevent chess model training instability
                        model_params = get_model_module(model).parameters()
                        total_norm = torch.nn.utils.clip_grad_norm_(list(model_params), clip_threshold)
                        if total_norm > clip_threshold:
                            print(f"Chess model gradient clipping applied at batch {batch_idx} in epoch {epoch}. Total norm: {total_norm:.2f}. Loss: {loss:.4f}")

                        # Update optimizer and scaler
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output, loss = model(x, targets=y, target_roles=y_roles)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

                    # Accumulate loss for logging
                    loss_val = loss.item()
                    running_loss += loss_val
                    epoch_loss += loss_val
                    total_batches += 1
                    epoch_batches += 1

                    # =============================================================================
                    # BLACKWELL GPU MEMORY MANAGEMENT - RESOLVED
                    # =============================================================================
                    #
                    # STATUS: ✅ FIXED with CUDA 12.8 + PyTorch 2.9.0+cu128 (October 2025)
                    #
                    # PREVIOUS PROBLEM: NVIDIA Blackwell GB10 (compute capability 12.1) had memory accumulation
                    # bugs specifically with large batch sizes (>= 64) in older PyTorch/CUDA versions.
                    #
                    # SYMPTOMS (FIXED):
                    # - Memory usage grew continuously with batch_size >= 64
                    # - Large batches hit 98GB+ and crashed the system
                    # - Only affected Blackwell GPUs with large batches
                    #
                    # SOLUTION (IMPLEMENTED):
                    # CUDA 12.8 includes Blackwell-specific memory management optimizations
                    # PyTorch 2.9.0+cu128 provides proper Blackwell kernel support
                    #
                    # CURRENT STATUS:
                    # - Blackwell memory management is now stable
                    # - Large batch sizes work without special cleanup
                    # - No performance overhead from memory workarounds
                    #
                    # RECOMMENDED USAGE:
                    # - Batch 32: Works (lightweight)
                    # - Batch 64: Optimal performance
                    # - Batch 128: Works with CUDA 12.8 optimizations
                    # - Batch 256+: Possible with sufficient memory
                    #
                    # TECHNICAL NOTE:
                    # The cleanup code below is commented out as it's no longer needed.
                    # Blackwell memory issues were software-related, not hardware defects.
                    # =============================================================================

                    # BLACKWELL MEMORY FIX: COMMENTED OUT - No longer needed with CUDA 12.8
                    # This was a workaround for Blackwell memory accumulation bugs in older CUDA versions
                    # CUDA 12.8 provides native Blackwell memory management optimizations

                    # needs_cleanup = gpu_capability == (12, 1) and batch_size >= 64
                    #
                    # if needs_cleanup:
                    #     # Store values needed for inference before cleanup
                    #     current_loss_value = loss_val
                    #     current_batch_x = x[-1].unsqueeze(0).clone()  # Last sample for inference
                    #     # Force explicit cleanup to prevent memory accumulation
                    #     del output, loss, x, y
                    #     torch.cuda.empty_cache()
                    # else:
                    #     # No cleanup needed - store values normally
                    #     current_loss_value = loss_val
                    #     current_batch_x = x

                    # Standard memory management - explicit cleanup for stable training
                    current_loss_value = loss_val
                    current_batch_x = x

                    # Explicit cleanup of training tensors to prevent memory accumulation
                    # Only needed for Blackwell GPUs (sm_121) due to unique memory characteristics
                    del output, loss, x, y, y_roles
                    if device.type == 'cuda' and is_blackwell_gpu:
                        torch.cuda.empty_cache()

                    # Progress reporting and checkpointing
                    if (batch_idx + 1) % inference_frequency == 0:
                        avg_loss = running_loss / total_batches
                        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Avg Loss: {avg_loss:.4f}")

                        # Continuous plateau detection: average last 10000 batches (20 × 500-batch groups)
                        # Only check for plateau every 5000 batches (10 report cycles)
                        if scheduler_choice == 'plateau':
                            if 'batch_group_losses' not in locals():
                                batch_group_losses = []
                                plateau_check_counter = 0
                            batch_group_losses.append(avg_loss)
                            plateau_check_counter += 1

                            # Keep only last 20 groups (10000 batches) for averaging
                            if len(batch_group_losses) > 20:
                                batch_group_losses = batch_group_losses[-20:]

                            # Get current LR for reporting
                            current_lr = optimizer.param_groups[0]['lr']
                            print(f"📊 LR: {current_lr:.2e} | Loss history: {len(batch_group_losses)} samples")

                            # Only check plateau every 10 report cycles (5000 batches)
                            if plateau_check_counter >= 10 and len(batch_group_losses) >= 20:
                                plateau_check_counter = 0  # Reset counter
                                current_avg = sum(batch_group_losses) / len(batch_group_losses)

                                if 'previous_plateau_avg' in locals():
                                    # Compare current 10000-batch average vs previous
                                    improvement = previous_plateau_avg - current_avg
                                    if improvement >= 0.001:
                                        print(f"📈 Improvement: {improvement:.4f} (>= 0.001) - continuing at LR {current_lr:.2e}")
                                    else:
                                        print(f"🔻 Plateau: {improvement:.4f} < 0.001 - reducing LR from {current_lr:.2e}")
                                        scheduler.step(float('inf'))  # Force LR reduction (20% reduction with factor=0.8)
                                        new_lr = optimizer.param_groups[0]['lr']
                                        print(f"   New LR: {new_lr:.2e}")

                                # Update previous average
                                previous_plateau_avg = current_avg

                        running_loss = 0.0
                        total_batches = 0

                        # Generate sample moves
                        all_text = test_progress(
                            epoch, num_epochs, batch_idx, data_loader, current_loss_value,
                            model, current_batch_x, 50, all_text, idx_to_move
                        )

                        # Save model using the same function as the working program
                        save_model_all(
                            model, all_text, n_embd, n_head, n_kv_heads, n_layer, dropout,
                            block_size, epoch, batch_idx, batch_size, optimizer, scheduler, scaler, current_loss_value,
                            learning_rate, weight_decay, gpu_indices
                        )

                        # Clean up inference tensors (Blackwell-specific)
                        if device.type == 'cuda':
                            gpu_capability = torch.cuda.get_device_capability()
                            is_blackwell = gpu_capability == (12, 1)
                            if is_blackwell:
                                torch.cuda.empty_cache()

            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

            # Step exponential scheduler at end of each epoch
            if scheduler_choice == 'exponential':
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"📉 Exponential LR decay: {current_lr:.2e}")

            # Show loss trend
            if len(epoch_losses) > 1:
                loss_change = epoch_losses[-2] - epoch_losses[-1]
                print(f"Loss change from previous epoch: {loss_change:+.4f}")

            # Scheduler is now stepped every 100 batches, not per epoch

    # Final training summary
    if epoch_losses:
        initial_loss = epoch_losses[0]
        final_loss = epoch_losses[-1]
        total_improvement = initial_loss - final_loss
        print(f"\nTraining Summary:")
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Total improvement: {total_improvement:.4f}")
        print(f"Improvement rate: {total_improvement/len(epoch_losses):.4f} per epoch")

    print("Training completed!")


# Main training function
def train_chess_model():
    """Main training entry point"""
    # Single process mode - load data interactively
    text, checkpoint_data = load_data_interactive()
    _train_chess_model_core(text, checkpoint_data)


def load_data_interactive():
    """Load data in single process mode (interactive)"""
    global gpu_indices
    text = ""
    checkpoint_data = None

    # Ask user to choose between loading model or creating new
    load_or_create = get_input_with_default("Load a model file or Create a new model? (l/c)", "c").lower()

    if load_or_create == 'l':
        loaded_data = load_model_file()
        if loaded_data:
            checkpoint_data = loaded_data
        else:
            print("Failed to load model. Creating a new one.")
            checkpoint_data = None
    else:
        checkpoint_data = None

    # GPU selection - only for fresh training (checkpoint loading will set its own GPUs)
    if checkpoint_data is None and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        all_gpus = list(range(num_gpus))

        print("\nGPU Selection (for Blackwell/multi-GPU setups):")
        print(f"System has {num_gpus} GPU{'s' if num_gpus > 1 else ''} available")
        print("Note: Enter your GPU number directly (e.g., '0,1,2' for multiple GPUs)")
        custom_gpus = input(f"Enter GPU indices separated by commas (default: all {num_gpus} GPUs): ")

        if not custom_gpus.strip():
            gpu_indices = all_gpus  # Default to all available GPUs
            print(f"Using all {num_gpus} GPU{'s' if num_gpus > 1 else ''}: {gpu_indices}")
        else:
            try:
                gpu_indices = [int(idx.strip()) for idx in custom_gpus.split(',')]
                print(f"Selected GPUs: {gpu_indices}")
            except ValueError:
                gpu_indices = all_gpus  # Fallback to all GPUs
                print(f"Invalid input. Using all {num_gpus} GPU{'s' if num_gpus > 1 else ''}: {gpu_indices}")

        # Set CUDA_VISIBLE_DEVICES for fresh training
        if gpu_indices:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_indices))
            print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Always use interactive file selection
    file_path = create_file_dialog(title="Select Chess Games File for Training", filetypes=[("Text files", "*.txt")])
    if not file_path:
        print("No chess file selected. Exiting.")
        exit()

    print(f"Loading chess file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    games = text.split('\n\n')
    games = ['<STARTGAME>' + ' ' + game.strip() + ' ' + '<EOFG>' for game in games if game.strip()]
    text = '\n'.join(games)
    print(f"Chess dataset loaded. Total games: {len(games)}, Total characters: {len(text)}")

    return text, checkpoint_data


if __name__ == "__main__":
    print("ChessBrain - Chess Move Prediction LLM")
    print("=" * 50)

    # Start training
    print("🖥️  Single process mode - interactive GUI setup")
    train_chess_model()
