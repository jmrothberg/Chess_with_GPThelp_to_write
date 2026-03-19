# ChessBrain Technical Reference

Detailed documentation of both tokenization modes, the per-game training strategy, data formats, and checkpoint structure.

## Supported Hardware
- **DGX Spark**: NVIDIA GB10 Blackwell (128GB unified memory)
- **Mac Studio**: Apple Silicon with MPS
- **Multi-GPU CUDA**: 4x A6000 Ada or similar

## Data Format

### Parquet Files (Preferred)

Training loads directly from parquet files with two required columns:

| Column | Type | Example |
|--------|------|---------|
| `Moves` | list/array of strings | `["e2e4", "e7e5", "g1f3", ...]` |
| `Result` | string | `"1-0"`, `"0-1"`, or `"1/2-1/2"` |

On load, each row is converted to the internal text format:
```
<W> E2E4 E7E5 G1F3 B8C6 ...
```

Results map to: `"1-0"` becomes `<W>` (white win), everything else becomes `<D>` (draw/black win).

Progress is displayed during conversion (loading ~2M games takes a few minutes):
```
Loading parquet file: chess_game_0001.parquet
Found 1,999,939 rows, converting to text...
Converting games: 47% (939,271/1,999,939)
```

### Text Files (Legacy)

Plain text format, one game per paragraph, double-newline separated:
```
<W> E2E4 E7E5 G1F3 B8C6 F1B5 A7A6 ...

<D> D2D4 D7D5 C2C4 E7E6 B1C3 ...
```

The converter script `parquettorext_withpromotion_Dec_12.py` is no longer needed but remains for reference.

## Tokenization Modes

At startup (new models only), the trainer prompts for tokenization mode. Loaded checkpoints auto-detect their mode.

### Classic Mode: 1 Token Per Move

Each UCI move maps to a single integer. The vocabulary is constructed from all possible chess moves:

```
move_id = from_square * 315 + to_offset * 5 + promo_index
```

Where:
- `from_square`: 0-63 (a8=0, h1=63)
- `to_offset`: 0-62 (destination square index, skipping the origin)
- `promo_index`: 0=none, 1=queen, 2=rook, 3=bishop, 4=knight

This gives 64 x 63 x 5 = **20,160 move tokens** plus 5 special tokens:

| Token | ID | Purpose |
|-------|-----|---------|
| `<STARTGAME>` | 20160 | Start of game boundary |
| `<EOFG>` | 20161 | End of game |
| `<PAD>` | 20162 | Padding for short games |
| `<W>` | 20163 | White wins |
| `<D>` | 20164 | Draw / Black wins |

**Total vocabulary: 20,165 tokens**

Examples of move-to-token mapping:
- `E2E4` (no promotion) -> single token
- `E7E8Q` (queen promotion) -> different single token
- `E7E8N` (knight underpromotion) -> yet another token

All 4 promotion types plus "no promotion" are fully represented as separate tokens.

**Model head**: Single `lm_head` linear layer with weight tying to the token embedding table. Standard next-token prediction.

**Default block_size**: 128 (128 half-moves = 64 full moves)

### 4-Token Mode: 4 Tokens Per Move

Each move is decomposed into 4 role-tagged sub-tokens:

```
e2e4 -> [WHITE, e2, e4, none]
e7e5 -> [BLACK, e7, e5, none]
a7a8q -> [WHITE, a7, a8, queen]
```

| Role | Vocabulary | Values |
|------|-----------|--------|
| COLOR | 2 | White, Black |
| FROM | 64 | a1-h8 |
| TO | 64 | a1-h8 |
| PROMO | 5 | none, queen, rook, bishop, knight |

Plus 5 special tokens shared across roles. **Total vocabulary: 140 tokens.**

**Model heads**: Four separate linear output heads, one per role. A role mask routes each token position to its correct head. The TO head receives an extra signal — the FROM square embedding is added to the hidden state, so destination prediction is conditioned on origin.

**Loss weighting**:

| Head | Weight | Rationale |
|------|--------|-----------|
| FROM | 1.0 | Core decision: which piece to move |
| TO | 1.0 | Core decision: where to move it |
| PROMO | 1.0 | Mostly "none" but must learn promotions |
| COLOR | 0.5 | Alternates trivially, downweighted to save gradient for FROM/TO |

**Default block_size**: 512 (512 / 4 = 128 half-moves = 64 full moves)

### Comparison

| | Classic | 4-Token |
|---|---------|---------|
| Vocabulary size | ~20K | 140 |
| Tokens per move | 1 | 4 |
| Default block_size | 128 | 512 |
| Game coverage | 64 full moves | 64 full moves |
| Output heads | 1 (lm_head) | 4 (color, from, to, promo) |
| Weight tying | Yes | No |
| FROM->TO conditioning | No (learned implicitly) | Yes (explicit embedding) |
| Forward passes per move (inference) | 1 | 4 |

## Per-Game Training Strategy

Both modes use the same per-game training strategy: every training sequence starts from `<STARTGAME>`.

### Why Per-Game Matters

**Random window approach (old)**: Grab any 256 tokens from any position in any game. The model often starts mid-game — it doesn't know how the pieces got there, whether castling is available, or whose advantage it is.

**Per-game approach (current)**: Every sequence starts at Move 1. The model builds board state in its hidden state step-by-step, learning cause and effect — why a piece is developed, how an opening leads to a middlegame plan.

### Padding Efficiency

Short games produce sequences with padding at the end:
```
[START] [e2e4] [c7c5] [g1f3] [d7d6] ... [PAD] [PAD] [PAD]
```

This is more efficient than it appears:
1. **Padding is masked**: The loss function ignores PAD tokens. Attention only sees real moves. Gradients are clean.
2. **Quality over quantity**: 100% of sequences are complete games. No broken mid-game fragments producing noisy gradients.
3. **No overlap**: 1 million moves across ~10,000 games = 10,000 sequences (vs ~1 million overlapping windows). An epoch runs ~100x faster.

## Checkpoint Format

Checkpoints are standard PyTorch `.pth` files containing:

```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'scaler_state_dict': ...,
    'hyperparameters': {
        'token_mode': 'classic' or '4token',
        'format_version': 3 (classic) or 2 (4-token),
        'vocab_size': 20165 or 140,
        'n_embd': ...,
        'n_head': ...,
        'n_kv_heads': ...,
        'block_size': ...,
        'n_layer': ...,
        'dropout': ...,
        ...
    },
    'tokenizer': { ... },  # Classic mode: full 20K dict; 4-token: 140-token dict
    'epoch': ...,
    'batch': ...,
}
```

The inference engine and GUI auto-detect `token_mode` from the checkpoint. Old checkpoints without `token_mode` default to `'4token'`.

## Inference Pipeline

`Chess_Inference_WB_2_12_26.py` loads any checkpoint and routes to the correct generation path:

**Classic mode**: Tokenize game history into single tokens, one forward pass, mask special tokens from output logits, return top-k moves by probability.

**4-Token mode**: Tokenize game history into role-tagged 4-token sequences, run 4 forward passes (COLOR -> FROM -> TO -> PROMO), combine probabilities, return top-k moves.

Both paths return a flat list of UCI strings (e.g., `['e2e4', 'g1f3', 'd2d4']`). The chess GUI validates each suggestion for legality and plays the first legal move.

## Ctrl+C Training Controls

Pressing Ctrl+C during training pauses after the current batch and offers:
- **Change learning rate**: Adjust without restarting
- **Load new data**: Select a new `.parquet` or `.txt` file. The dataset is rebuilt while preserving the optimizer state.
- **Quit**: Save checkpoint and exit
