"""
ChessBrain Inference Engine

Specialized inference engine for chess move prediction using transformer models.
Supports coordinate notation chess moves with optimized MobileLLM architecture.

Key Features:
- Chess move tokenization (coordinate notation)
- Support for both basic and optimized model architectures
- Game boundary masking for proper chess game handling (MobileLLM)
- Top-k sampling for diverse move generation
- Integration with ChessBrain training system

Supported Architectures:
- TransformerModel: Standard GPT-style transformer
- MobileLLMModel: Memory-efficient with RMSNorm, MultiQueryAttention, SwiGLU

Usage:
- ChessBrain Integration: Called by training scripts for progress monitoring
- API: Use generate_response() function programmatically for chess moves

Version History:
- Sep 24, 2024: Added separate chess moves tokenizer
- Sep 26, 2024: Mac compatibility and latest file selection
- Nov 20, 2024: MobileLLM architecture integration
- Nov 22, 2024: Support for both basic and optimized chess models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob

# Device configuration for inference
# For performance: CUDA > MPS > CPU
# For debugging/development: Force CPU to avoid GPU memory issues
device = torch.device('cpu')

# Global state for chess inference API usage
# These persist across function calls for efficiency
global_model = None          # Loaded MobileLLM chess model
global_tokenizer = None      # Chess move tokenizer (dict)
global_tokenizer_reverse = None # Reverse mapping for move decoding
global_use_characters = False  # Chess-only: no character-level tokenization
global_use_chess_moves = True  # Chess-only: use coordinate notation

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

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('causal_mask', torch.tril(torch.ones(1024, 1024)))
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

        # Repeat keys and values to match the number of query heads
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        if self.flash_available:
            # Prepare masks
            causal_mask = self.causal_mask[:T, :T].bool()  # Shape: [T, T]
            if mask is not None:
                game_mask = mask[:, :T, :T].bool()  # Shape: [B, T, T]
                combined_mask = torch.logical_and(
                    causal_mask.unsqueeze(0),  # Shape: [1, T, T]
                    game_mask  # Shape: [B, T, T]
                )  # Resulting shape: [B, T, T]
            else:
                combined_mask = causal_mask.unsqueeze(0)  # Shape: [1, T, T]

            # Unsqueeze to add the num_heads dimension
            attention_mask = combined_mask.unsqueeze(1)  # Shape: [B, 1, T, T]

            # Use flash attention with the correctly shaped mask
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,  # Shape: [B, 1, T, T]
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False  # We're handling causality in our mask
            )
        else:
            # Traditional attention (fallback if flash attention is unavailable)
            # Compute attention scores
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Apply causal masking
            causal_mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, T, T]
            causal_mask = causal_mask.expand(B, self.n_heads, T, T)  # Expand to batch and heads
            att = att.masked_fill(causal_mask == 0, float('-inf'))

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




# === Role-specific 4-token-per-ply constants (must match Chess_Brain_WB_2_12_26.py) ===
ROLE_COLOR = 0; ROLE_FROM = 1; ROLE_TO = 2; ROLE_PROMO = 3; ROLE_SPECIAL = -1
COLOR_OFFSET = 0; FROM_OFFSET = 2; TO_OFFSET = 66; PROMO_OFFSET = 130
STARTGAME = 135; EOFG = 136; PAD = 137; W_RESULT = 138; D_RESULT = 139
ROLE_VOCAB_SIZE = 140


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
    m = {}
    m['<WHITE>'] = 0; m['<BLACK>'] = 1
    for sq in range(64):
        m[f'F:{square_to_uci(sq)}'] = FROM_OFFSET + sq
        m[f'T:{square_to_uci(sq)}'] = TO_OFFSET + sq
    for i, l in enumerate(['none', 'q', 'r', 'b', 'n']):
        m[f'<PROMO:{l}>'] = PROMO_OFFSET + i
    m['<STARTGAME>'] = STARTGAME; m['<EOFG>'] = EOFG; m['<PAD>'] = PAD
    m['<W>'] = W_RESULT; m['<D>'] = D_RESULT
    return m

move_to_idx = create_move_to_idx()



def load_model_file(frommain=False):
    """
    Load and initialize a trained chess transformer model from checkpoint.

    Automatically detects model architecture from checkpoint metadata.
    Supports both basic and optimized chess move prediction models.

    Model Detection Logic:
    1. Load checkpoint and examine hyperparameters
    2. Check for MobileLLM-specific layers (RMSNorm, SwiGLU, etc.)
    3. Load appropriate chess model architecture
    4. Load chess tokenizer and model weights
    5. Handle DataParallel/torch.compile prefix handling

    Supported Architectures:
    - TransformerModel: Standard GPT-style transformer for chess
    - MobileLLMModel: Memory-efficient with RMSNorm, MultiQueryAttention, SwiGLU

    Args:
        frommain: Internal flag for main script execution

    Returns:
        Tuple: (model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, tokenizer)

    Raises:
        File selection dialog if model_file_path not provided and not frommain
    """
    running_on_mac = os.name == 'posix' and os.uname().sysname == 'Darwin'
    try:
        if not frommain and running_on_mac:
            chess_directory = "."
            if not os.path.exists(chess_directory):
                print(f"Directory not found: {chess_directory}")
                return None, None, None, None, None, None, None, None

            pth_files = glob.glob(os.path.join(chess_directory, "*.pth"))
            if not pth_files:
                print(f"No .pth files found in {chess_directory}")
                return None, None, None, None, None, None, None, None

            # Sort by modification time (newest first)
            pth_files.sort(key=os.path.getctime, reverse=True)

            print("\nAvailable model files (newest first):")
            for i, file_path in enumerate(pth_files, 1):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                mod_time = os.path.getctime(file_path)
                from datetime import datetime
                mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                print(f"{i}. {os.path.basename(file_path)} ({file_size:.1f} MB, {mod_time_str})")

            while True:
                try:
                    choice = input("\nEnter the number of the model file to load (or 'q' to quit): ").strip()
                    if choice.lower() == 'q':
                        return None, None, None, None, None, None, None, None
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(pth_files):
                        model_file = pth_files[choice_idx]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(pth_files)}")
                except ValueError:
                    print("Please enter a valid number or 'q' to quit")
        else:
            import tkinter as tk
            from tkinter import filedialog
            model_file = filedialog.askopenfilename(filetypes=[("PyTorch Model Files", "*.pth")])

        if model_file:
            checkpoint = torch.load(model_file, map_location="cpu")
            hyperparameters = checkpoint['hyperparameters']
            state_dict = checkpoint['model_state_dict']
            
            # Extract hyperparameters with fallbacks
            vocab_size = hyperparameters['vocab_size']
            n_embd = hyperparameters['n_embd']
            n_head = hyperparameters['n_head']
            n_layer = hyperparameters['n_layer']
            dropout = hyperparameters['dropout']
            block_size = hyperparameters['block_size']
            
            # Determine model architecture based on checkpoint contents
            fmt_version = hyperparameters.get('format_version', 1)
            has_role_heads = any('head_color' in key for key in state_dict.keys())
            has_factorized_heads = any('from_head' in key for key in state_dict.keys())
            has_mobile_llm_features = any('rms_1' in key or 'swiglu' in key for key in state_dict.keys())

            if has_role_heads or fmt_version >= 2:
                # New 4-token-per-ply format with role-specific heads
                print("Loading ChessModel (role-specific heads, format v2)...")
                n_kv_heads = hyperparameters.get('n_kv_heads', n_head // 4)

                import sys
                sys.path.append('/home/jonathan/Chess')
                from Chess_Brain_WB_2_12_26 import ChessModel

                model = ChessModel(
                    vocab_size=vocab_size,
                    n_embd=n_embd,
                    n_head=n_head,
                    n_kv_heads=n_kv_heads,
                    block_size=block_size,
                    n_layer=n_layer,
                    dropout=dropout,
                    use_chess=True
                )

                # Use the 140-token tokenizer
                tokenizer = create_move_to_idx()

            elif has_factorized_heads:
                # Old factorized heads (v1) - not compatible
                print("ERROR: Old factorized-head checkpoint not compatible. Re-train with Chess_Brain_WB_2_12_26.")
                return None, None, None, None, None, None, None, None

            elif has_mobile_llm_features:
                # Medium-old checkpoint with MobileLLM features - use MobileLLMModel
                print("Loading MobileLLMModel (chess-optimized)...")
                n_kv_heads = hyperparameters.get('n_kv_heads', n_head // 4)
                model = MobileLLMModel(
                    vocab_size=vocab_size,
                    n_embd=n_embd,
                    n_head=n_head,
                    n_kv_heads=n_kv_heads,
                    block_size=block_size,
                    n_layer=n_layer,
                    dropout=dropout,
                    use_chess=True
                )
                tokenizer = checkpoint.get('tokenizer')

            else:
                # Very old checkpoint - use basic TransformerModel
                print("Loading TransformerModel (basic)...")
                model = TransformerModel(
                    vocab_size=vocab_size,
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    n_layer=n_layer,
                    dropout=dropout,
                    use_chess=True
                )
                tokenizer = checkpoint.get('tokenizer')

            # Clean state dict keys if needed
            # Handle common wrapper prefixes (DataParallel, torch.compile, etc.)
            cleaned_sd = {}
            for k, v in state_dict.items():
                nk = k
                # DataParallel prefix
                if nk.startswith('module.'):
                    nk = nk[len('module.'):]
                # torch.compile (with or without DataParallel)
                if nk.startswith('_orig_mod.module.'):
                    nk = nk[len('_orig_mod.module.'):]
                elif nk.startswith('_orig_mod.'):
                    nk = nk[len('_orig_mod.'):]
                cleaned_sd[nk] = v

            state_dict = cleaned_sd
            
            # Try to load state dict, with error handling
            try:
                model.load_state_dict(state_dict)
                print("Model loaded successfully!")
            except RuntimeError as e:
                print(f"Error loading state dict: {e}")
                print("Attempting to load with strict=False...")
                model.load_state_dict(state_dict, strict=False)
            
            return model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, tokenizer
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None, None, None, None


def generate_response(model, tokenizer, tokenizer_reverse, input_text,
                     tokens_to_generate=5, top_k=10, use_characters=False, use_chess_moves=True, use_dna=False):
    """
    Generate top-k candidate NEXT moves using 4-token-per-ply role-specific heads.

    Returns a list of up to top_k UCI move strings (e.g. ['e2e4', 'g1f3', ...])
    ranked by model confidence. The chess game checks legality and picks the first legal one.

    Args:
        model: ChessModel with role-specific heads
        tokenizer: 140-token vocab dict
        tokenizer_reverse: Reverse mapping
        input_text: Game history string (e.g. "<STARTGAME> e2e4 e7e5 ...")
        tokens_to_generate: Unused (kept for API compatibility)
        top_k: Number of candidate moves to return (default: 10)
        use_characters/use_chess_moves/use_dna: Unused (API compatibility)

    Returns:
        List of top-k UCI move strings (e.g. ['e2e4', 'g1f3', 'd2d4', ...])
    """
    model.eval()
    model.to(device)

    if tokenizer is None:
        print("Error: Tokenizer not available")
        return []

    # Tokenize game history into 4-token-per-ply format
    tokens = []
    ply = 0
    i = 0
    while i < len(input_text):
        if input_text[i:i+11] == '<STARTGAME>':
            tokens.append(STARTGAME); ply = 0; i += 11
        elif input_text[i:i+6] == '<EOFG>':
            tokens.append(EOFG); i += 6
        elif input_text[i:i+3] == '<W>':
            tokens.append(W_RESULT); i += 3
        elif input_text[i:i+3] == '<D>':
            tokens.append(D_RESULT); i += 3
        elif input_text[i].isspace():
            i += 1
        elif i + 4 <= len(input_text):
            move_str = None
            if i + 5 <= len(input_text) and input_text[i+4].isalpha() and input_text[i+4].lower() in 'qrbn':
                c = input_text[i:i+5]
                if c[0].isalpha() and c[1].isdigit() and c[2].isalpha() and c[3].isdigit():
                    move_str = c; i += 5
            if move_str is None:
                c = input_text[i:i+4]
                if c[0].isalpha() and c[1].isdigit() and c[2].isalpha() and c[3].isdigit():
                    move_str = c; i += 4
                else:
                    i += 1; continue
            is_white = (ply % 2 == 0)
            ct, ft, tt, pt = parse_uci_move(move_str, is_white)
            tokens.extend([ct, ft, tt, pt])
            ply += 1
        else:
            i += 1

    # Truncate to block_size
    block_size = model.block_size if hasattr(model, 'block_size') else 512
    if len(tokens) > block_size:
        tokens = tokens[-block_size:]

    input_seq = torch.tensor([tokens], dtype=torch.long).to(device)

    # Handle torch.compile wrapper for direct attribute access
    model_raw = model._orig_mod if hasattr(model, '_orig_mod') else model

    with torch.no_grad():
        # Step 1: Predict COLOR (we know what it should be, but let model confirm)
        is_white = (ply % 2 == 0)
        color_tok = COLOR_OFFSET + (0 if is_white else 1)
        input_seq = torch.cat([input_seq, torch.tensor([[color_tok]], device=device)], dim=1)
        if input_seq.shape[1] > block_size:
            input_seq = input_seq[:, -block_size:]

        # Step 2: Get FROM probabilities
        output, _ = model(input_seq)
        from_logits = output['from'][0, -1]  # [64]
        from_probs = F.softmax(from_logits, dim=-1)

        # Get top-k FROM squares
        num_from = min(top_k, 64)
        top_from_probs, top_from_sqs = torch.topk(from_probs, k=num_from)

        # Step 3: For each candidate FROM, predict TO (conditioned on FROM)
        candidates = []  # (score, from_sq, to_sq, promo_idx)

        for fi in range(num_from):
            from_sq = top_from_sqs[fi].item()
            from_prob = top_from_probs[fi].item()
            from_tok = FROM_OFFSET + from_sq

            # Append FROM token and run forward
            seq_with_from = torch.cat([input_seq, torch.tensor([[from_tok]], device=device)], dim=1)
            if seq_with_from.shape[1] > block_size:
                seq_with_from = seq_with_from[:, -block_size:]

            output2, _ = model(seq_with_from)
            h_last = output2['hidden'][0, -1]

            # Condition on FROM for TO prediction
            from_emb = model_raw.emb_from(torch.tensor(from_sq, device=device))
            h_conditioned = h_last + from_emb
            to_logits = model_raw.head_to(h_conditioned)
            to_logits[from_sq] = float('-inf')  # TO != FROM
            to_probs = F.softmax(to_logits, dim=-1)

            # Top TO squares for this FROM
            num_to = min(3, 64)
            top_to_probs, top_to_sqs = torch.topk(to_probs, k=num_to)

            for ti in range(num_to):
                to_sq = top_to_sqs[ti].item()
                to_prob = top_to_probs[ti].item()
                score = from_prob * to_prob

                # Check if promotion (pawn reaching back rank)
                from_rank = 8 - (from_sq // 8)
                to_rank = 8 - (to_sq // 8)
                is_promo = (is_white and from_rank == 7 and to_rank == 8) or \
                           (not is_white and from_rank == 2 and to_rank == 1)

                if is_promo:
                    # Get promo prediction
                    to_tok = TO_OFFSET + to_sq
                    seq_with_to = torch.cat([seq_with_from,
                                             torch.tensor([[to_tok]], device=device)], dim=1)
                    if seq_with_to.shape[1] > block_size:
                        seq_with_to = seq_with_to[:, -block_size:]
                    output3, _ = model(seq_with_to)
                    promo_logits = output3['promo'][0, -1]
                    promo_idx = promo_logits.argmax(dim=-1).item()
                    if promo_idx == 0:
                        promo_idx = 1  # Default to queen if model says none
                else:
                    promo_idx = 0

                candidates.append((score, from_sq, to_sq, promo_idx))

        # Sort by score (descending) and return top-k UCI strings
        candidates.sort(key=lambda x: x[0], reverse=True)
        promo_chars = ['', 'q', 'r', 'b', 'n']

        result_moves = []
        seen = set()
        for score, from_sq, to_sq, promo_idx in candidates:
            uci = square_to_uci(from_sq) + square_to_uci(to_sq)
            if promo_idx > 0 and promo_idx < len(promo_chars):
                uci += promo_chars[promo_idx]
            if uci not in seen:
                seen.add(uci)
                result_moves.append(uci)
                if len(result_moves) >= top_k:
                    break

        print(f"Top {len(result_moves)} candidate moves: {result_moves}")
        return result_moves

# Chess Inference API:
# generate_response():
#   - Top-k sampling for diverse chess move generation
#   - Chess coordinate notation tokenization
#   - Returns list of possible move continuations
#   - Used for: ChessBrain integration, move prediction API


def initialize_model():
    """
    Initialize global model state for chess inference API usage.

    Loads a chess model and sets up global variables for repeated inference calls.
    Provides programmatic access to chess move generation without reloading.

    Global State Set:
    - global_model: Loaded MobileLLM chess model
    - global_tokenizer: Chess move tokenizer (dict)
    - global_tokenizer_reverse: Reverse mapping for move decoding

    Returns:
        Loaded model instance, or None if loading failed

    Usage:
        initialize_model()  # Load once
        # Then use global_model for multiple chess inferences
    """
    global global_model, global_tokenizer, global_tokenizer_reverse

    # Load chess model
    model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, tokenizer = load_model_file()
    if model is None:
        print("Failed to load chess model.")
        return None

    # Set global state for chess API usage
    global_model = model
    global_tokenizer = tokenizer
    if tokenizer is not None:
        global_tokenizer_reverse = {v: k for k, v in tokenizer.items()}
    else:
        print("Warning: Tokenizer is None, cannot create reverse mapping")
        global_tokenizer_reverse = None

    return model

