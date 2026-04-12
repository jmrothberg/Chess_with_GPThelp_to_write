# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A transformer-based language model that learns to play chess by reading raw move sequences from Stockfish self-play games. No chess engine, no search tree, no handcrafted rules — just a GPT-style decoder transformer trained on millions of games. Author: Jonathan M. Rothberg.

## Commands

```bash
# Stay current with upstream (preferred before large work sessions)
git pull origin main

# Install dependencies
pip install -r chess_requirement.txt

# Train a model (interactive prompts for mode, data, GPUs)
python Chess_Brain_3_21_26.py

# Play chess via Pygame GUI (human / CPU Search / Neural per side)
python Chess_4_8_26.py

# Plot training loss from checkpoint filenames
python plot_loss_March_20_26.py [checkpoint_folder]

# Combine .txt game files (archived copy under OLD chess brains/)
python "OLD chess brains/combine_chess_datasets.py" <group_size> <input_dir> <output_dir>
```

There is no test suite. Validation is done through GUI play, move legality tracking, and loss curves.

## Architecture

### Three Main Files

- **`Chess_Brain_3_21_26.py`** — Training script. Model definition, both dataset classes, data loading, full training loop with multi-GPU support.
- **`Chess_Inference.py`** — Inference engine. Auto-detects checkpoint mode (classic vs 4-token) and provides `generate_response()` for move generation.
- **`Chess_4_8_26.py`** — Pygame GUI. Imports `Chess_Inference` as `brain_inference`. Human play, classical CPU **Search** (alpha-beta; optional Stockfish), and **Neural** (LLM) per side.

Older dated copies of training/data utilities live under **`OLD chess brains/`** for reference only.

### Two Tokenization Modes (selected at training startup)

**Classic Mode (1 token per move):** ~20K vocab (64×63×5 move tokens + 5 special). Block size 128. Single `lm_head` output (weight-tied to embeddings). One forward pass per move.

**4-Token Mode (4 tokens per move):** 140 vocab (2 COLOR + 64 FROM + 64 TO + 5 PROMO + 5 special). Block size 512. Four role-specific output heads. Four sequential forward passes per move, with FROM→TO conditioning via `emb_from` embedding.

### Transformer Backbone (shared by both modes)

- Grouped Query Attention (GQA): 8 query heads, 2 KV heads (4:1 ratio)
- RMSNorm (pre-norm), SwiGLU FFN, absolute positional embeddings
- Defaults: n_embd=512, n_layer=12, dropout=0.0, batch_size=512
- Game boundary masking: attention cannot cross `<STARTGAME>` tokens
- Gradient checkpointing for memory efficiency

### Key Classes in Brain (training script)

| Class | Purpose |
|-------|---------|
| `RMSNorm` | Layer normalization (faster than LayerNorm) |
| `MultiQueryAttention` | GQA with flash attention support |
| `SwiGLU` | Gated FFN activation |
| `ChessBlock` | Single transformer block (pre-norm → attn → residual → pre-norm → FFN → residual) |
| `ChessModel` | Top-level model: embeddings + blocks + mode-specific output heads |
| `ChessMovesDataset` | 4-token mode dataset (returns x, y, y_roles) |
| `ClassicChessMovesDataset` | Classic mode dataset (y_roles is all -1 sentinel) |

### Data Pipeline

Input: Parquet files (columns: `Moves` as list of UCI strings, `Result` as "1-0"/"0-1"/"1/2-1/2") or plain text (one game per paragraph). Result mapping: "1-0" → `<W>`, everything else → `<D>`. Each game is wrapped as `<STARTGAME> <W/D> moves <EOFG>`. Per-game training: every sequence starts at move 1 (no mid-game fragments).

### Inference Flow (4-token mode)

1. Append deterministic COLOR token (white=0, black=1 based on ply)
2. Forward pass → top-k FROM squares
3. For each FROM candidate: append FROM token, forward pass, condition hidden state via `emb_from`, predict TO
4. If promotion-eligible (back rank): predict PROMO piece
5. Score = from_prob × to_prob × promo_prob, return top-k UCI moves

### Checkpoint Format

Saved as `.pth` files. Filename encodes hyperparameters: `C{layers}H{heads}E{embd}_B{batch}_E{epoch}B{batch}_L{loss}_{timestamp}.pth`. Contains model/optimizer/scheduler state dicts plus `hyperparameters` dict with `token_mode` field for auto-detection.

### Training Controls

Ctrl+C during training pauses after current batch and presents a menu: change learning rate, load new data (preserves optimizer state), or quit (saves checkpoint). Optimizer: Adafactor. Mixed precision (FP16) with GradScaler. Gradient clipping at max_norm=5.0.

## Key Conventions

- Loss weights in 4-token mode: FROM=1.0, TO=1.0, PROMO=1.0, COLOR=0.5 (downweighted because trivially predictable)
- The model suggests moves; legality is validated externally by the GUI's chess engine
- Square indexing: a8=0, h1=63 (formula: `(8 - rank) * 8 + file_idx`)
- `y_roles = -1` sentinel means classic mode (skip role-specific heads, use `lm_head`)
- Result markers (`<W>`, `<D>`) are metadata only — no outcome prediction heads
