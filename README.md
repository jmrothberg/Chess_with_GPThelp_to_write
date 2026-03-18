# Chess AI — Learn Chess from Move Sequences

A transformer that learns to play chess by reading raw move sequences. No chess engine, no search tree, no handcrafted rules — just a language model trained on millions of Stockfish games.

## How It Works

The model reads chess games written as sequences of UCI moves:

```
<STARTGAME> e2e4 e7e5 g1f3 b8c6 f1b5 ... <W>
```

Each move (like `e2e4`) is broken into **4 tokens**:

| Token | What it means | Vocabulary |
|-------|--------------|------------|
| **COLOR** | Whose turn (White or Black) | 2 values |
| **FROM** | Square the piece moves from | 64 squares |
| **TO** | Square the piece moves to | 64 squares |
| **PROMO** | Promotion piece (if pawn reaches last rank) | 5 values (none, q, r, b, n) |

So the game `e2e4 e7e5` becomes:

```
WHITE e2 e4 none  BLACK e7 e5 none  ...
```

Total vocabulary: **140 tokens** (not 20,000+ like one-token-per-move approaches).

### Four Output Heads

The transformer has a shared body (attention layers) and four small output heads, one per token role:

```
                    ┌─► head_color (2 classes)
Transformer Body ───┼─► head_from  (64 classes)
                    ├─► head_to    (64 classes)  ◄── conditioned on FROM
                    └─► head_promo (5 classes)
```

Each head only predicts its own token type. The TO head gets an extra signal: the FROM square embedding is added to the hidden state, so "where to go" is directly conditioned on "where from."

### Loss Weighting

Not all predictions are equally important:

| Head | Weight | Why |
|------|--------|-----|
| FROM | 1.0 | The core chess decision — which piece to move |
| TO | 1.0 | The core chess decision — where to move it |
| PROMO | 1.0 | Mostly "none," easy to learn |
| COLOR | 0.5 | Trivially predictable (alternates W/B), downweighted so it doesn't steal gradient from FROM/TO |

### Generation

To generate a move, the model runs 4 forward passes:
1. Predict COLOR (always deterministic, but model confirms)
2. Predict FROM square (which piece to move)
3. Predict TO square (where to move it, conditioned on FROM)
4. Predict PROMO (promotion piece, usually "none")

The 4 tokens are decoded back to UCI notation (e.g., `e2e4`).

## Architecture

- **Type**: Decoder-only transformer (GPT-style)
- **Attention**: Grouped-Query Attention (fewer KV heads than query heads)
- **Normalization**: RMSNorm (pre-norm)
- **FFN**: SwiGLU (3 weight matrices per layer)
- **Context**: 512 tokens = ~128 half-moves of game history

Current model: 24 layers, 16 heads, 4 KV heads, embed 1024 (~365M parameters).

## Project Files

| File | Purpose |
|------|---------|
| `Chess_Brain_WB_2_12_26.py` | Model training (multi-GPU) |
| `Chess_Inference_WB_2_12_26.py` | Model inference |
| `Chess_WB_2_12_26.py` | Interactive chess game (Pygame GUI) |
| `plot_loss_Nov_9_25.py` | Training loss visualization |
| `parquettorext_withpromotion_Dec_12.py` | Convert parquet data to UCI text |
| `combine_chess_datasets.py` | Combine game files for training |

### Older Model (single-token-per-move)

`ChessBrain_Multi_per_game_torchcompile_12_13_25.py` is the previous version that tokenized each move as a single token from a ~20K vocabulary. The new 4-token approach replaces it with a 140-token vocabulary and role-specific heads.

## Training

### Data
- Source: Stockfish self-play games in UCI format
- Format: One game per paragraph, moves space-separated
- Each game wrapped with `<STARTGAME>` and result token (`<W>`, `<D>`)

### Quick Start
```bash
pip install -r chess_requirement.txt
python Chess_Brain_WB_2_12_26.py
```

Training supports Ctrl+C to pause and change learning rate or load new data without losing optimizer state.

### Hardware
Tested on 4x NVIDIA RTX 6000 Ada (48GB each). Automatic batch size estimation based on actual free GPU memory.

## Author

**Jonathan M. Rothberg** — [@jmrothberg](https://github.com/jmrothberg)

## License

MIT License
