"""
ChessBrain Inference Engine — Chess_Inference.py (Sept 20, 2026)

Loads `.pth` checkpoints from Chess_Brain_mp_spawn_9_20_26.py (and older trainers)
and generates UCI moves for Chess_9_20_26.py. No dependency on the training script.

CURRENT FORMAT (result-aware, Sept 20, 2026) — what changed vs older inference:
- Classic (1 token/move) OR 4-token (COLOR/FROM/TO/PROMO) auto-detected from checkpoint
- Vocab extended: `<B>` (black won) and `<U>` (unknown result) — 4-token vocab 142;
  classic specials grow after the fixed 20,160 move tokens (old ids unchanged)
- Prompt side: GUI builds `<STARTGAME> <W|B> moves…` so Neural plays like the winner
  (`build_history_prompt` / RESULT_TOKEN_FOR_SIDE)
- Optional value head: 3-way W/D/B; GUI may re-rank legal candidates via `rerank_by_value`
- Optional packed_positions: position ids restart at each `<STARTGAME>` (game packing)
- Device: CPU by default (DGX checkpoint → Mac/Linux play). Override with CHESS_DEVICE=mps|cuda

Still supported (legacy):
- Older ChessModel checkpoints without value head / packing
- Very old TransformerModel / MobileLLMModel weights (kept for load compatibility)

Architecture (current ChessModel): RMSNorm, MultiQueryAttention (GQA), SwiGLU, game masks.

API: initialize_model / generate_response / generate_candidates / build_history_prompt /
     rerank_by_value — used by the Pygame GUI and by the trainer’s sample dumps.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# Device configuration for inference
# For performance: CUDA > MPS > CPU
# Sept 20, 2026: CPU by default so a checkpoint trained on the DGX plays on any Mac with just
# `pip install torch pygame`. Set CHESS_DEVICE=mps (Apple GPU) or CHESS_DEVICE=cuda to opt in.
def _select_inference_device():
    want = os.environ.get('CHESS_DEVICE', 'cpu').lower()
    if want == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    if want == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if want not in ('cpu', 'mps', 'cuda'):
        print(f"CHESS_DEVICE={want!r} not recognised, using cpu")
    return torch.device('cpu')

device = _select_inference_device()

# Global state for chess inference API usage
# These persist across function calls for efficiency
global_model = None          # Loaded MobileLLM chess model
global_tokenizer = None      # Chess move tokenizer (dict)
global_tokenizer_reverse = None # Reverse mapping for move decoding
global_use_characters = False  # Chess-only: no character-level tokenization
global_use_chess_moves = True  # Chess-only: use coordinate notation

# =============================================================================================
# LEGACY CHECKPOINT SUPPORT
# TransformerModel / MobileLLMModel (and their building blocks) are kept so that very old
# checkpoints still load. Current checkpoints use ChessModel further below. RMSNorm,
# MultiQueryAttention, SwiGLU and Block are shared with ChessModel.
# =============================================================================================

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention for autoregressive transformers.

    Implements the classic attention mechanism from "Attention is All You Need"
    with causal masking for autoregressive generation. Used in the basic
    TransformerModel architecture.

    Key Features:
    - Multi-head attention for capturing different attention patterns
    - Causal masking to prevent attending to future tokens
    - Stores attention weights for visualization/debugging
    - Dropout on attention weights for regularization

    Note: This is the basic implementation. For optimized variants, see:
    - MultiQueryAttention (memory efficient)
    - RoPEMultiHeadAttention (position-aware)

    Args:
        n_embd: Embedding dimension (must be divisible by n_head)
        n_head: Number of attention heads
        block_size: Maximum sequence length for causal masking
        dropout: Dropout probability for attention weights
    """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.n_head = n_head
        self.head_size = head_size

        # Linear projections for Q, K, V
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # Causal mask to prevent attending to future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Attention dropout and output projection
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape

        # Generate Q, K, V and reshape for multi-head attention
        # Shape: (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Compute attention scores and apply causal masking
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Apply attention to values and reshape back
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        y = self.proj(y)

        # Store attention for visualization (used by analysis functions)
        self.last_attention = att
        return y

class FeedForward(nn.Module):
    """
    Standard transformer feed-forward network.

    Expands input dimension by 4x with a linear layer, applies ReLU activation,
    then contracts back to original dimension. Includes dropout for regularization.

    This is the basic implementation used in TransformerModel. For optimized
    variants, see SwiGLU in MobileLLM architectures.

    Architecture:
    - Expansion: n_embd → 4*n_embd (capacity for complex patterns)
    - Activation: ReLU (non-linearity)
    - Contraction: 4*n_embd → n_embd (back to model dimension)
    - Regularization: Dropout on output

    Args:
        n_embd: Input/output embedding dimension
        dropout: Dropout probability applied after final projection
    """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expansion layer
            nn.ReLU(),                      # Non-linearity
            nn.Linear(4 * n_embd, n_embd), # Contraction layer
            nn.Dropout(dropout),           # Regularization
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Standard transformer decoder block with pre-layer normalization.

    Implements the classic transformer architecture with attention followed by
    feed-forward, using residual connections and layer normalization. This is
    the basic building block used in TransformerModel.

    Architecture (GPT-style):
    - Pre-layer norm on input
    - Multi-head self-attention with residual connection
    - Pre-layer norm on attention output
    - Feed-forward network with residual connection

    Key Features:
    - Stores attention weights and activations for visualization
    - Pre-norm architecture (modern transformer design)
    - Used in basic TransformerModel (not optimized variants)

    Note: For optimized variants, see:
    - ChessBlock (with RMSNorm, MultiQueryAttention, SwiGLU)

    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        block_size: Maximum sequence length
        dropout: Dropout probability for regularization
    """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)  # Pre-attention normalization
        self.ln2 = nn.LayerNorm(n_embd)  # Pre-feedforward normalization

    def forward(self, x):
        # Attention block with residual connection
        x = x + self.sa(self.ln1(x))

        # Store attention for visualization (used by analysis functions)
        self.last_attention = self.sa.last_attention

        # Feed-forward block with residual connection
        x = x + self.ffwd(self.ln2(x))

        # Store activation for visualization
        self.last_activation = x

        return x

class TransformerModel(nn.Module):
    """
    Standard GPT-style transformer model for chess move prediction.

    Implements the basic transformer decoder architecture adapted for chess.
    Used for basic chess move generation tasks.

    Architecture:
    - Token embeddings + positional embeddings
    - Stack of TransformerBlock layers
    - Final layer normalization
    - Language modeling head (logits projection)

    Key Features:
    - Autoregressive generation (predicts next chess move given previous)
    - Causal attention masking throughout
    - Optimized for chess coordinate notation

    Args:
        vocab_size: Size of the chess move vocabulary
        n_embd: Embedding dimension (model width)
        n_head: Number of attention heads per layer
        block_size: Maximum sequence length
        n_layer: Number of transformer blocks
        dropout: Dropout probability for regularization
    """
    def __init__(self, vocab_size, n_embd, n_head, block_size, n_layer, dropout):
        super().__init__()

        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        # Final normalization and output head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token + position embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization and logits
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided (training mode)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    A more efficient alternative to standard LayerNorm that normalizes by root mean square
    instead of mean and variance. Used in MobileLLM architectures for improved efficiency.

    Key Advantages over LayerNorm:
    - ~18% faster inference (fewer operations, no mean calculation)
    - Better gradient flow in deep networks
    - Equivalent or better performance than LayerNorm
    - Simpler computation: RMS = sqrt(mean(x²))

    Formula: RMSNorm(x) = (x / RMS(x)) * γ
    where RMS(x) = sqrt(mean(x²) + ε)

    Used in: MobileLLMModel, ChessModel (chess-optimized architectures)

    Args:
        dim: Feature dimension to normalize
        eps: Small epsilon for numerical stability (default: 1e-5)
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scaling parameter

    def forward(self, x):
        # Compute RMS normalization: x / sqrt(mean(x²) + ε)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight
class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention with shared Key-Value heads for memory efficiency.

    An optimized attention mechanism where multiple query heads share the same key and value
    heads, reducing memory footprint while maintaining attention quality. Particularly effective
    for chess models where memory efficiency is critical.

    Key Advantages:
    - Reduced VRAM usage (especially for large models)
    - Faster computation due to fewer KV operations
    - Maintains attention quality for complex pattern recognition
    - Causal masking for autoregressive generation

    Architecture:
    - Multiple query heads (n_head) for diverse attention patterns
    - Shared KV heads (n_kv_heads) to reduce memory/compute
    - Typical ratio: 4:1 (n_head=8, n_kv_heads=2)

    Used in: MobileLLMModel, ChessModel (memory-constrained architectures)

    Args:
        n_embd: Embedding dimension (must be divisible by n_head)
        n_head: Number of query heads (attention outputs)
        n_kv_heads: Number of shared key/value heads (memory bottleneck)
        dropout: Dropout probability for attention weights
    """
    def __init__(self, n_embd, n_head, n_kv_heads, dropout):
        super().__init__()
        head_dim = n_embd // n_head
        self.n_heads = n_head
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(n_embd, n_head * head_dim)
        self.kv_proj = nn.Linear(n_embd, n_kv_heads * head_dim * 2)  # Combine k and v projections
        self.out_proj = nn.Linear(n_embd, n_embd)

        # QK-Norm: stabilizes attention logits, prevents explosion
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        self.dropout = nn.Dropout(dropout)
        # Causal mask is built from T each forward (fixed 1024 buffer broke block_size 1536)
        self.flash_available = hasattr(F, 'scaled_dot_product_attention')
        if self.flash_available:
            print("Using Flash Attention in MultiQueryAttention")

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # Project queries
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)

        # Project keys and values together
        kv = self.kv_proj(x).view(B, T, self.n_kv_heads, 2, self.head_dim)  # (B, T, n_kv_heads, 2, head_dim)
        kv = kv.transpose(1, 2)  # (B, n_kv_heads, T, 2, head_dim)
        k, v = kv[..., 0, :], kv[..., 1, :]  # Split into k and v

        # QK-Norm: normalize Q and K before attention to prevent logit explosion
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Repeat keys and values to match the number of query heads
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        if self.flash_available:
            if mask is not None:
                causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
                combined_mask = causal.unsqueeze(0) & mask[:, :T, :T].bool()
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=combined_mask.unsqueeze(1),
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False
                )
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=True
                )
        else:
            # Traditional attention (fallback if flash attention is unavailable)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            causal = torch.ones(T, T, device=x.device).tril()
            att = att.masked_fill(causal == 0, float('-inf'))

            # Apply additional mask if provided
            if mask is not None:
                game_mask = mask.unsqueeze(1)  # Shape: [B, 1, T, T]
                game_mask = game_mask.expand(B, self.n_heads, T, T)  # Expand to heads
                att = att.masked_fill(game_mask == 0, float('-inf'))

            # Apply softmax and dropout
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)

            # Apply attention to values
            y = att @ v

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)

        return y
    

class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function
    - More efficient than ReLU/GELU
    - Better performance for language tasks
    - Uses gating mechanism to control information flow
    """
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4  # 4x multiplier as per paper
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        gate = F.silu(self.w1(x))  # SiLU activation for gating
        hidden = self.w2(x)
        return self.w3(gate * hidden)

class Block(nn.Module):
    """
    Optimized Transformer Block combining all MobileLLM improvements:
    1. RMSNorm for faster normalization
    2. Multi-Query Attention for efficient attention
    3. SwiGLU for better activation
    """
    def __init__(self, n_embd, n_head, n_kv_heads, dropout):  # Added dropout parameter
        super().__init__()
        # Pre-normalization (better training stability)
        self.rms_1 = RMSNorm(n_embd)
        # Multi-Query Attention with shared KV heads
        self.attn = MultiQueryAttention(
            n_embd=n_embd, 
            n_head=n_head,
            n_kv_heads=n_kv_heads,  # Pass through exactly what we want
            dropout=dropout
        )
        # Second normalization
        self.rms_2 = RMSNorm(n_embd)
        # SwiGLU feedforward
        self.swiglu = SwiGLU(n_embd)
        self.dropout = nn.Dropout(dropout)  # Now dropout is passed in

    def forward(self, x, mask=None):
        # Attention with pre-norm
        x = x + self.dropout(self.attn(self.rms_1(x), mask=mask))
        # FFN with pre-norm
        x = x + self.dropout(self.swiglu(self.rms_2(x)))
        return x

class MobileLLMModel(nn.Module):
    """
    Main model incorporating MobileLLM optimizations while maintaining chess game support
    """
    def __init__(self, vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, use_chess=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.use_chess = use_chess
        if use_chess:
            self.start_game_token = move_to_idx['<STARTGAME>']
        
        # Standard embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Use MultiQueryAttention with separate Q and KV head counts
        self.blocks = nn.ModuleList([
            Block(
                n_embd=n_embd, 
                n_head=n_head,
                n_kv_heads=n_kv_heads,  # Pass through our desired ratio
                dropout=dropout
            ) for _ in range(n_layer)
        ])
        
        # Final RMSNorm instead of LayerNorm
        self.rms_final = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_game_mask(self, idx):
        if not self.use_chess:
            return None
        mask = torch.ones_like(idx, dtype=torch.float32)
        game_boundaries = (idx == self.start_game_token).float().cumsum(dim=1)
        mask = (game_boundaries.unsqueeze(1) == game_boundaries.unsqueeze(2)).float()
        return mask

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Get embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Apply transformer blocks with chess game mask
        for block in self.blocks:
            x = block(x, mask=self.create_game_mask(idx))
            
        # Final normalization and prediction
        x = self.rms_final(x)
        logits = self.lm_head(x)
        
        # Calculate loss if training
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss




# ChessBlock is identical to Block at inference time (gradient checkpointing is training-only)
ChessBlock = Block


# =============================================================================================
# CURRENT CHECKPOINT FORMAT (Chess_Brain_mp_spawn_9_20_26.py, result-aware, Sept 20, 2026)
# Vocab / special-token constants. Must match the training script exactly.
# =============================================================================================
# === Role-specific 4-token-per-ply constants ===
ROLE_COLOR = 0; ROLE_FROM = 1; ROLE_TO = 2; ROLE_PROMO = 3; ROLE_SPECIAL = -1
COLOR_OFFSET = 0; FROM_OFFSET = 2; TO_OFFSET = 66; PROMO_OFFSET = 130
STARTGAME = 135; EOFG = 136; PAD = 137; W_RESULT = 138; D_RESULT = 139
B_RESULT = 140   # <B> black won (Sept 20, 2026; 0-1 used to be lumped into <D>)
U_RESULT = 141   # <U> unknown result -> value head must predict the outcome
ROLE_VOCAB_SIZE = 142

# Special tokens shared by both modes. Classic ids are fixed offsets after the 20,160 move tokens,
# so <STARTGAME>..<D> match old 20,165-token checkpoints and <B>,<U> extend them.
CLASSIC_SPECIAL_TOKENS = ['<STARTGAME>', '<EOFG>', '<PAD>', '<W>', '<D>', '<B>', '<U>']
CLASSIC_MOVE_TOKENS = 64 * 63 * 5
RESULT_TOKEN_FOR_SIDE = {'W': '<W>', 'B': '<B>'}   # "play like the winner" prompt token per side
VALUE_CLASS = {'W': 0, 'D': 1, 'B': 2}             # value head output order


def classic_special_ids():
    return {name: CLASSIC_MOVE_TOKENS + i for i, name in enumerate(CLASSIC_SPECIAL_TOKENS)}


def role_special_ids():
    return {'<STARTGAME>': STARTGAME, '<EOFG>': EOFG, '<PAD>': PAD,
            '<W>': W_RESULT, '<D>': D_RESULT, '<B>': B_RESULT, '<U>': U_RESULT}


class ChessModel(nn.Module):
    """
    Chess move prediction transformer (inference-only copy, no Chess_Brain dependency).
    Supports both classic (single lm_head) and 4-token (role-specific heads) modes.

    Sept 20, 2026 additions (must mirror Chess_Brain_mp_spawn_9_20_26.py):
    - packed_positions: position ids restart at every <STARTGAME> (models trained with game packing)
    - use_value_head: extra 3-way head predicting the game result (W/D/B) from the moves so far
    Older checkpoints without these still load (flags false / head absent).
    """
    def __init__(self, vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout,
                 use_chess=True, use_dna=False, token_mode='4token',
                 use_value_head=False, packed_positions=False, start_game_token=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.use_chess = use_chess
        self.token_mode = token_mode
        self.use_value_head = use_value_head
        self.packed_positions = packed_positions
        self.special_ids = classic_special_ids() if token_mode == 'classic' else role_special_ids()
        self.start_game_token = self.special_ids['<STARTGAME>'] if start_game_token is None else start_game_token

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.register_buffer('pos_indices', torch.arange(block_size))

        self.blocks = nn.ModuleList([
            ChessBlock(n_embd=n_embd, n_head=n_head, n_kv_heads=n_kv_heads, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.rms_final = RMSNorm(n_embd)

        if token_mode == 'classic':
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding_table.weight
        else:
            self.head_color = nn.Linear(n_embd, 2)
            self.head_from = nn.Linear(n_embd, 64)
            self.head_to = nn.Linear(n_embd, 64)
            self.head_promo = nn.Linear(n_embd, 5)
            self.emb_from = nn.Embedding(64, n_embd)

        if use_value_head:
            self.head_value = nn.Linear(n_embd, 3)   # logits over (W, D, B)

    def create_game_mask(self, idx):
        if not self.use_chess or self.start_game_token is None:
            return None
        game_boundaries = (idx == self.start_game_token).float().cumsum(dim=1)
        return (game_boundaries.unsqueeze(1) == game_boundaries.unsqueeze(2)).float()

    def game_start_index(self, idx):
        """Index of the most recent <STARTGAME> at or before each position (0 if none)."""
        B, T = idx.shape
        t = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        return torch.where(idx == self.start_game_token, t, torch.zeros_like(t)).cummax(dim=1).values

    def _backbone(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        if self.packed_positions and self.use_chess:
            pos_ids = (self.pos_indices[:T].unsqueeze(0) - self.game_start_index(idx)).clamp(0, self.block_size - 1)
            pos_emb = self.position_embedding_table(pos_ids)
        else:
            pos_emb = self.position_embedding_table(self.pos_indices[:T])
        x = tok_emb + pos_emb
        game_mask = self.create_game_mask(idx)
        for block in self.blocks:
            x = block(x, mask=game_mask)
        return self.rms_final(x)

    def forward(self, idx, targets=None, target_roles=None):
        x = self._backbone(idx)
        if self.token_mode == 'classic':
            return self.lm_head(x), None
        out = {'hidden': x, 'color': self.head_color(x), 'from': self.head_from(x), 'promo': self.head_promo(x)}
        if self.use_value_head:
            out['value'] = self.head_value(x)
        return out, None

    @torch.no_grad()
    def predict_value(self, idx):
        """Softmax over (W, D, B) at the last position of each row. Prompt should start <STARTGAME> <U>."""
        if not self.use_value_head:
            return None
        h = self._backbone(idx)
        return F.softmax(self.head_value(h[:, -1]), dim=-1)


def create_classic_move_to_idx():
    """Create classic ~20K vocab: 64*63*5 move tokens + special tokens (7 since Sept 20, 2026: +<B>,<U>)."""
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
    for idx, token in enumerate(CLASSIC_SPECIAL_TOKENS, start=len(m)):
        m[token] = idx
    return m


def create_classic_idx_to_move(classic_move_to_idx):
    """Reverse mapping for classic tokenizer."""
    return {idx: move for move, idx in classic_move_to_idx.items()}


def uci_to_square(file_char, rank_char):
    file_idx = ord(file_char.lower()) - ord('a')
    rank_idx = int(rank_char)
    return (8 - rank_idx) * 8 + file_idx


def square_to_uci(sq):
    return chr(ord('a') + (sq % 8)) + str(8 - (sq // 8))


def parse_uci_move(move_str, is_white):
    move_str = move_str.lower().strip()
    from_sq = uci_to_square(move_str[0], move_str[1])
    to_sq = uci_to_square(move_str[2], move_str[3])
    promo_map = {'q': 1, 'r': 2, 'b': 3, 'n': 4}
    promo_idx = promo_map.get(move_str[4], 0) if len(move_str) >= 5 else 0
    return (COLOR_OFFSET + (0 if is_white else 1),
            FROM_OFFSET + from_sq, TO_OFFSET + to_sq, PROMO_OFFSET + promo_idx)


def create_move_to_idx():
    """4-token vocab (142 tokens since Sept 20, 2026: added <B> and <U>)."""
    m = {}
    m['<WHITE>'] = 0; m['<BLACK>'] = 1
    for sq in range(64):
        m[f'F:{square_to_uci(sq)}'] = FROM_OFFSET + sq
        m[f'T:{square_to_uci(sq)}'] = TO_OFFSET + sq
    for i, l in enumerate(['none', 'q', 'r', 'b', 'n']):
        m[f'<PROMO:{l}>'] = PROMO_OFFSET + i
    m.update(role_special_ids())
    return m

move_to_idx = create_move_to_idx()


def _clean_state_dict_keys(state_dict):
    """Strip DataParallel / torch.compile wrapper prefixes from checkpoint keys."""
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('_orig_mod.module.'):
            nk = nk[len('_orig_mod.module.'):]
        elif nk.startswith('_orig_mod.'):
            nk = nk[len('_orig_mod.'):]
        cleaned[nk] = v
    return cleaned


def load_model_file(checkpoint_path=None):
    """
    Load and initialize a trained chess transformer model from checkpoint.

    Everything the GUI needs is read from the checkpoint and attached to the model object
    (so two models of different modes can be loaded for White and Black at the same time):
        model._token_mode        'classic' | '4token'
        model._tokenizer         name -> id dict actually used for training
        model._tokenizer_reverse id -> name
        model._block_size
        model._has_value_head    True when the checkpoint has the W/D/B value head
        model._special_ids       special-token ids for this mode

    Checkpoint hyperparameters honoured (old checkpoints simply lack the new keys):
        vocab_size, n_embd, n_head, n_kv_heads, n_layer, dropout, block_size, token_mode,
        has_value_head (default False), packed_positions (default False)

    Args:
        checkpoint_path: Path to a .pth checkpoint (required; no GUI dialog in this module).

    Returns:
        Tuple: (model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, tokenizer)
        or eight Nones on failure.
    """
    NONE8 = (None,) * 8
    try:
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}" if checkpoint_path else "load_model_file: checkpoint_path is required.")
            return NONE8

        # map_location='cpu' keeps DGX-trained checkpoints loadable on a Mac
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        hyperparameters = checkpoint['hyperparameters']
        state_dict = _clean_state_dict_keys(checkpoint['model_state_dict'])

        vocab_size = hyperparameters['vocab_size']
        n_embd = hyperparameters['n_embd']
        n_head = hyperparameters['n_head']
        n_layer = hyperparameters['n_layer']
        dropout = hyperparameters['dropout']
        block_size = hyperparameters['block_size']
        n_kv_heads = hyperparameters.get('n_kv_heads', n_head // 4)

        fmt_version = hyperparameters.get('format_version', 1)
        token_mode = hyperparameters.get('token_mode', '4token')
        has_role_heads = any('head_color' in key for key in state_dict)
        has_lm_head = any('lm_head' in key for key in state_dict)
        has_factorized_heads = any('from_head' in key for key in state_dict)
        has_mobile_llm_features = any('rms_1' in key or 'swiglu' in key for key in state_dict)
        # Sept 20, 2026 fields; fall back to inspecting weights for checkpoints saved without them
        has_value_head = bool(hyperparameters.get('has_value_head', 'head_value.weight' in state_dict))
        packed_positions = bool(hyperparameters.get('packed_positions', False))

        tokenizer = checkpoint.get('tokenizer') if isinstance(checkpoint.get('tokenizer'), dict) else None

        if token_mode == 'classic' or (fmt_version >= 3 and has_lm_head and has_mobile_llm_features):
            token_mode = 'classic'
            if tokenizer is None:
                tokenizer = create_classic_move_to_idx()
            print(f"Loading ChessModel (classic 1-token mode, vocab={vocab_size}, value_head={has_value_head}, packed={packed_positions})...")
            model = ChessModel(vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout,
                               use_chess=True, token_mode='classic',
                               use_value_head=has_value_head, packed_positions=packed_positions,
                               start_game_token=tokenizer.get('<STARTGAME>'))

        elif has_role_heads or fmt_version >= 2:
            token_mode = '4token'
            if tokenizer is None:
                # Old 4-token checkpoints (140 tokens) saved no tokenizer: rebuild and trim to vocab_size
                tokenizer = {k: v for k, v in create_move_to_idx().items() if v < vocab_size}
            print(f"Loading ChessModel (4-token mode, vocab={vocab_size}, value_head={has_value_head}, packed={packed_positions})...")
            model = ChessModel(vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout,
                               use_chess=True, token_mode='4token',
                               use_value_head=has_value_head, packed_positions=packed_positions)

        elif has_factorized_heads:
            print("ERROR: Old factorized-head checkpoint not compatible. Re-train with the current training script.")
            return NONE8

        elif has_mobile_llm_features:
            # Legacy: MobileLLMModel checkpoint
            print("Loading MobileLLMModel (legacy)...")
            token_mode = 'classic'
            if tokenizer is not None:
                global move_to_idx
                move_to_idx = tokenizer   # MobileLLMModel.__init__ reads the module global
            model = MobileLLMModel(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_kv_heads=n_kv_heads,
                                   block_size=block_size, n_layer=n_layer, dropout=dropout, use_chess=True)

        else:
            # Legacy: basic TransformerModel checkpoint
            print("Loading TransformerModel (legacy)...")
            token_mode = 'classic'
            model = TransformerModel(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head,
                                     block_size=block_size, n_layer=n_layer, dropout=dropout)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"State dict: missing={missing} unexpected={unexpected}")
        else:
            print("Model loaded successfully!")

        # Attach per-model info for the GUI / generation helpers
        model._token_mode = token_mode
        model._tokenizer = tokenizer
        model._tokenizer_reverse = {v: k for k, v in tokenizer.items()} if tokenizer else None
        model._block_size = block_size
        model._has_value_head = bool(has_value_head and isinstance(model, ChessModel))
        model._special_ids = getattr(model, 'special_ids', None) or (
            {n: tokenizer[n] for n in CLASSIC_SPECIAL_TOKENS if tokenizer and n in tokenizer})
        model.eval()
        model.to(device)
        print(f"Model token mode: {token_mode}, block_size: {block_size}, device: {device}")

        return model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, tokenizer

    except Exception as e:
        print(f"An error occurred loading the checkpoint: {e}")
        return NONE8


# =============================================================================================
# Prompt building and tokenisation (shared by classic and 4-token generation)
# =============================================================================================
def _model_info(model, tokenizer=None):
    """(token_mode, tokenizer, block_size) for a loaded model, falling back to legacy globals."""
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    token_mode = getattr(model, '_token_mode', None) or getattr(raw, 'token_mode', '4token')
    tok = getattr(model, '_tokenizer', None) or tokenizer or global_tokenizer
    if tok is None:
        tok = create_classic_move_to_idx() if token_mode == 'classic' else create_move_to_idx()
    block_size = getattr(model, '_block_size', None) or getattr(raw, 'block_size', 512)
    return token_mode, tok, block_size


def result_token_for_side(model, side):
    """'<W>' for White / '<B>' for Black if this model's vocab has it (old models: <B> missing -> None)."""
    _, tok, _ = _model_info(model)
    name = RESULT_TOKEN_FOR_SIDE.get(side)
    return name if name and name in tok else None


def build_history_prompt(model, moves, side, result_token=None, reserve_tokens=None):
    """
    Build the text prompt the model sees: '<STARTGAME> <W|B> e2e4 e7e5 ...'.

    The header (<STARTGAME> + result token) is always kept; only the OLDEST moves are dropped
    when the game is too long for block_size. `reserve_tokens` leaves room for the move being
    generated (defaults to one move: 1 classic token / 4 role tokens).

    Args:
        model:        loaded model (attributes from load_model_file)
        moves:        list of UCI strings played so far (any case, no special tokens)
        side:         'W' or 'B' — the side about to move (the neural player)
        result_token: override the header token, e.g. '<U>' for value probing
    """
    token_mode, tok, block_size = _model_info(model)
    per_move = 1 if token_mode == 'classic' else 4
    if reserve_tokens is None:
        reserve_tokens = per_move
    header = ['<STARTGAME>']
    rt = result_token if result_token is not None else result_token_for_side(model, side)
    if rt and rt in tok:
        header.append(rt)
    max_moves = max(0, (block_size - len(header) - reserve_tokens) // per_move)
    kept = [m for m in moves if m and not m.startswith('<')]
    if len(kept) > max_moves:
        kept = kept[-max_moves:]
    return ' '.join(header + kept)


def _tokenize_history(input_text, tokenizer, token_mode):
    """
    Tokenise a game history string for either mode.

    Returns (tokens, ply). Special tokens <STARTGAME> <EOFG> <PAD> <W> <D> <B> <U> are looked up in
    the tokenizer (classic) or the role constants (4-token); ones the model does not know are skipped.
    """
    tokens = []
    ply = 0
    if token_mode == 'classic':
        specials = tokenizer
    else:
        # Only special tokens this model's vocab knows (an old 140-token model has no <B>/<U>)
        specials = {n: i for n, i in role_special_ids().items() if not tokenizer or n in tokenizer}
    i, n = 0, len(input_text)
    while i < n:
        ch = input_text[i]
        if ch == '<':
            close = input_text.find('>', i)
            if close == -1:
                break
            name = input_text[i:close + 1]
            if name == '<STARTGAME>':
                ply = 0
            if name in specials:
                tokens.append(specials[name])
            i = close + 1
            continue
        if ch.isspace():
            i += 1
            continue
        # UCI move: 4 chars, optional promotion letter
        move_str = None
        if i + 5 <= n and input_text[i + 4].lower() in 'qrbn' and input_text[i + 4].isalpha():
            c = input_text[i:i + 5]
            if c[0].isalpha() and c[1].isdigit() and c[2].isalpha() and c[3].isdigit():
                move_str = c
        if move_str is None and i + 4 <= n:
            c = input_text[i:i + 4]
            if c[0].isalpha() and c[1].isdigit() and c[2].isalpha() and c[3].isdigit():
                move_str = c
        if move_str is None:
            i += 1
            continue
        i += len(move_str)
        if token_mode == 'classic':
            tid = tokenizer.get(move_str.upper())
            if tid is not None:
                tokens.append(tid)
        else:
            tokens.extend(parse_uci_move(move_str, ply % 2 == 0))
        ply += 1
    return tokens, ply


# =============================================================================================
# Move generation
# =============================================================================================
@torch.no_grad()
def generate_candidates(model, input_text, top_k=10):
    """
    Top-k candidate NEXT moves with their model probability.

    Returns a list of (uci_lowercase, prob) sorted best first. Legality is NOT checked here;
    the GUI filters against its own legal-move list.
    """
    token_mode, tok, block_size = _model_info(model)
    model.eval()
    model.to(device)
    tokens, ply = _tokenize_history(input_text, tok, token_mode)
    if not tokens:
        return []
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model

    if token_mode == 'classic':
        tokens = tokens[-block_size:]
        logits, _ = model(torch.tensor([tokens], dtype=torch.long, device=device))
        next_logits = logits[0, -1].float()
        special_ids = {tok[n] for n in CLASSIC_SPECIAL_TOKENS if n in tok}
        for sid in special_ids:
            if sid < next_logits.shape[0]:
                next_logits[sid] = float('-inf')
        probs = F.softmax(next_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[0]))
        rev = getattr(model, '_tokenizer_reverse', None) or {v: k for k, v in tok.items()}
        out = []
        for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
            name = rev.get(tid, '')
            if name and not name.startswith('<'):
                out.append((name.lower(), p))
        return out

    # ---- 4-token mode: COLOR -> FROM -> (per FROM) TO -> PROMO ----
    is_white = (ply % 2 == 0)
    seq = tokens + [COLOR_OFFSET + (0 if is_white else 1)]
    seq = seq[-block_size:]
    input_seq = torch.tensor([seq], dtype=torch.long, device=device)

    output, _ = model(input_seq)
    from_probs = F.softmax(output['from'][0, -1].float(), dim=-1)
    num_from = min(top_k, 64)
    top_from_probs, top_from_sqs = torch.topk(from_probs, k=num_from)

    # One batched forward for all FROM candidates (same length)
    from_sqs = top_from_sqs.tolist()
    batch = torch.cat([input_seq.expand(num_from, -1),
                       (FROM_OFFSET + top_from_sqs).unsqueeze(1)], dim=1)
    if batch.shape[1] > block_size:
        batch = batch[:, -block_size:]
    output2, _ = model(batch)
    h_last = output2['hidden'][:, -1]                                   # [K, n_embd]
    from_emb = raw.emb_from(torch.tensor(from_sqs, device=device))      # [K, n_embd]
    to_logits = raw.head_to(h_last + from_emb).float()                  # [K, 64]
    to_logits[torch.arange(num_from), torch.tensor(from_sqs, device=device)] = float('-inf')  # TO != FROM
    to_probs = F.softmax(to_logits, dim=-1)

    candidates = []   # (score, from_sq, to_sq, promo_idx)
    promo_needed = []  # (index into candidates, seq)
    for fi in range(num_from):
        from_sq = from_sqs[fi]
        from_prob = top_from_probs[fi].item()
        top_to_probs, top_to_sqs = torch.topk(to_probs[fi], k=3)
        for to_prob, to_sq in zip(top_to_probs.tolist(), top_to_sqs.tolist()):
            from_rank = 8 - (from_sq // 8)
            to_rank = 8 - (to_sq // 8)
            is_promo = (is_white and from_rank == 7 and to_rank == 8) or (not is_white and from_rank == 2 and to_rank == 1)
            candidates.append([from_prob * to_prob, from_sq, to_sq, 0])
            if is_promo:
                promo_needed.append((len(candidates) - 1, seq + [FROM_OFFSET + from_sq, TO_OFFSET + to_sq]))

    if promo_needed:
        pb = torch.tensor([s[-block_size:] for _, s in promo_needed], dtype=torch.long, device=device)
        output3, _ = model(pb)
        promo_idx = output3['promo'][:, -1].argmax(dim=-1).tolist()
        for (ci, _), pidx in zip(promo_needed, promo_idx):
            candidates[ci][3] = pidx if pidx > 0 else 1   # never "none" on a promotion square

    candidates.sort(key=lambda c: c[0], reverse=True)
    promo_chars = ['', 'q', 'r', 'b', 'n']
    out, seen = [], set()
    for score, from_sq, to_sq, pidx in candidates:
        uci = square_to_uci(from_sq) + square_to_uci(to_sq) + promo_chars[pidx]
        if uci not in seen:
            seen.add(uci)
            out.append((uci, score))
            if len(out) >= top_k:
                break
    return out


def generate_response(model, tokenizer, tokenizer_reverse, input_text,
                      tokens_to_generate=5, top_k=10, use_characters=False, use_chess_moves=True, use_dna=False):
    """
    Top-k candidate NEXT moves as plain UCI strings (kept for API compatibility).

    Routes to classic or 4-token generation from the model's attributes.
    `tokenizer` / `tokenizer_reverse` are only used if the model has no attached tokenizer.
    `tokens_to_generate`, `use_characters`, `use_chess_moves`, `use_dna` are unused.
    """
    if getattr(model, '_tokenizer', None) is None and tokenizer is not None:
        model._tokenizer = tokenizer
        model._tokenizer_reverse = tokenizer_reverse or {v: k for k, v in tokenizer.items()}
    cands = generate_candidates(model, input_text, top_k=top_k)
    moves = [uci for uci, _ in cands]
    print(f"Top {len(moves)} candidate moves: {moves}")
    return moves


@torch.no_grad()
def rerank_by_value(model, moves, side, candidate_ucis):
    """
    Score legal candidate moves with the value head.

    For each candidate, the prompt is '<STARTGAME> <U> moves... candidate' (result hidden with
    <U> so the head has to judge the position) and the score is P(side wins) - P(side loses).
    Returns [(uci, value_score)] in the order given, or [] if the model has no value head.
    """
    if not getattr(model, '_has_value_head', False) or not candidate_ucis:
        return []
    token_mode, tok, block_size = _model_info(model)
    if '<U>' not in tok:
        return []
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    prompt = build_history_prompt(model, moves, side, result_token='<U>')
    base, ply = _tokenize_history(prompt, tok, token_mode)
    rows = []
    for uci in candidate_ucis:
        if token_mode == 'classic':
            tid = tok.get(uci.upper())
            if tid is None:
                return []   # candidate outside the vocab (should not happen for legal UCI)
            rows.append((base + [tid])[-block_size:])
        else:
            rows.append((base + list(parse_uci_move(uci, ply % 2 == 0)))[-block_size:])
    probs = raw.predict_value(torch.tensor(rows, dtype=torch.long, device=device))   # [K, 3]
    win_col, lose_col = (0, 2) if side == 'W' else (2, 0)
    scores = (probs[:, win_col] - probs[:, lose_col]).tolist()
    return list(zip(candidate_ucis, scores))


# Chess Inference API:
# generate_candidates(): top-k (uci, prob) for the next move
# generate_response():   same, plain UCI list (legacy signature)
# rerank_by_value():     value-head score per candidate (needs a checkpoint with has_value_head)
# build_history_prompt(): '<STARTGAME> <W|B> moves...' with block-size-safe truncation


def initialize_model(checkpoint_path=None):
    """
    Initialize global model state for chess inference API usage.

    Loads a chess model and sets up global variables for repeated inference calls.
    Provides programmatic access to chess move generation without reloading.

    Global State Set:
    - global_model: Loaded chess model
    - global_tokenizer: Chess move tokenizer (dict)
    - global_tokenizer_reverse: Reverse mapping for move decoding

    Note: the GUI should prefer the per-model attributes (model._tokenizer, model._block_size,
    model._token_mode) because the globals are overwritten each time a model is loaded.

    Args:
        checkpoint_path: Path to a .pth file. If None, returns None (caller must supply a path).

    Returns:
        Loaded model instance, or None if loading failed
    """
    global global_model, global_tokenizer, global_tokenizer_reverse

    if not checkpoint_path:
        print("initialize_model: checkpoint_path is required.")
        return None

    model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, tokenizer = load_model_file(
        checkpoint_path=checkpoint_path
    )
    if model is None:
        print("Failed to load chess model.")
        return None

    global_model = model
    global_tokenizer = tokenizer
    if tokenizer is not None:
        global_tokenizer_reverse = {v: k for k, v in tokenizer.items()}
    else:
        print("Warning: Tokenizer is None, cannot create reverse mapping")
        global_tokenizer_reverse = None

    return model
