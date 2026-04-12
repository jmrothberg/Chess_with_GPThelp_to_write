# Chess AI — Learn Chess from Move Sequences

A transformer that learns to play chess by reading raw move sequences. Training uses **no chess engine, no search tree, no handcrafted rules** — only next-move prediction on millions of Stockfish games.

The **Pygame client** (`Chess_4_8_26.py`) adds a separate **classical CPU search** (alpha-beta, transposition table, quiescence). That search does **not** call the neural net. You can play with **Search** on one side and **Neural** (transformer) on the other, or any mix.

## How It Works

The model reads chess games as sequences of UCI moves and learns to predict the next move. Training data is loaded directly from **parquet files** containing Stockfish self-play games — no preprocessing required.

```
<STARTGAME> <W> e2e4 e7e5 g1f3 b8c6 f1b5 ... <EOFG>
```

Two tokenization modes are available, selected at training startup:

### Classic Mode (1 token per move)

Each UCI move is a single token from a ~20K vocabulary. The vocabulary covers all possible moves: 64 origin squares x 63 destination squares x 5 promotion options (none, queen, rook, bishop, knight) = 20,160 move tokens + 5 special tokens.

- **Vocabulary**: ~20K tokens
- **Default context**: 128 tokens = 128 half-moves (64 full moves)
- **Loss**: Standard cross-entropy with weight-tied lm_head
- **Generation**: Single forward pass, top-k from output logits

### 4-Token Mode (4 tokens per move)

Each move is decomposed into 4 sub-tokens with role-specific output heads:

| Token | What it means | Vocabulary |
|-------|--------------|------------|
| **COLOR** | Whose turn | 2 values |
| **FROM** | Origin square | 64 squares |
| **TO** | Destination square | 64 squares |
| **PROMO** | Promotion piece | 5 values (none, q, r, b, n) |

- **Vocabulary**: 140 tokens
- **Default context**: 512 tokens = 128 half-moves (64 full moves)
- **Loss**: Weighted sum across 4 role-specific heads
- **Generation**: 4 sequential forward passes (one per token role)

Both modes default to the same game coverage (~64 full moves of context).

## Architecture

- **Type**: Decoder-only transformer (GPT-style)
- **Attention**: Grouped-Query Attention (fewer KV heads than query heads)
- **Normalization**: RMSNorm (pre-norm)
- **FFN**: SwiGLU (3 weight matrices per layer)
- **Shared backbone**: Identical transformer body for both tokenization modes

See [README_CHESS_PER_GAME.md](README_CHESS_PER_GAME.md) for detailed documentation of both modes, the per-game training strategy, and parquet format details.

## Keeping this repo current (preferred)

The canonical tree lives on GitHub (`main`). **Preferred:** from your clone root, run `git pull origin main` whenever you want the latest training, inference, and GUI scripts. That keeps filenames and behavior aligned with [Chess_with_GPThelp_to_write](https://github.com/jmrothberg/Chess_with_GPThelp_to_write). Commit or stash local edits first if `git pull` reports conflicts.

## Project Files

| File / location | Purpose |
|-----------------|---------|
| `Chess_Brain_3_21_26.py` | Model training (multi-GPU, both token modes) |
| `Chess_Inference.py` | Neural inference (auto-detects classic vs 4-token from checkpoint) |
| `Chess_4_8_26.py` | Pygame GUI: human / **Search** (CPU) / **Neural** (LLM) per side |
| `Chess_LLM_models/` | **Local** folder for `*.pth` checkpoints (directory tracked; weight files not in git) |
| `plot_loss_March_20_26.py` | Training loss visualization |
| `chess_requirement.txt` | Python dependencies for this repo |
| `OLD chess brains/` | Archived older training / data scripts (superseded filenames, kept for reference) |

## `Chess_LLM_models/` (checkpoints)

Clone the repo, then **copy** your `.pth` files into `Chess_LLM_models/` (or set env `CHESS_LLM_DIR` to another folder). Git **does not** store those weights; only the empty folder placeholder is committed so everyone has the same path.

## `OLD chess brains/`

Older copies of dataset/training utilities (e.g. per-game torchcompile brain, parquet converter, combine script) live here for reference. Active training is expected to use `Chess_Brain_3_21_26.py` and the current inference/game files above.

## Training

### Data
- **Source**: Stockfish self-play games
- **Format**: Parquet files with `Moves` (list of UCI strings) and `Result` columns, or plain text (one game per paragraph)
- **Loading**: Parquet files are read directly in the training script

### Quick Start
```bash
pip install -r chess_requirement.txt
python Chess_Brain_3_21_26.py
```

At startup, choose:
1. **New model or load checkpoint** — existing checkpoints auto-detect their mode
2. **Token mode** (new models only) — Classic (recommended) or 4-Token
3. **Training data** — select a `.parquet` or `.txt` file

Training supports Ctrl+C to pause and change learning rate or load new data without losing optimizer state.

### Hardware
Tested on 4x NVIDIA RTX 6000 Ada (48GB each). Automatic batch size estimation based on actual free GPU memory.

## Playing (GUI)

```bash
python Chess_4_8_26.py
```

Use **a** / **z** to cycle each side between **Search** (classical minimax) and **Neural** (transformer). **W** / **B** pick a `.pth` for that side. The GUI validates moves with its chess rules; the neural path suggests UCI strings and Search uses its own eval.

Optional: `CHESS_USE_STOCKFISH=1` (and `CHESS_STOCKFISH` path) can delegate **Search** to Stockfish if installed — see `Chess_4_8_26.py` for UCI integration.

## Author

**Jonathan M. Rothberg** — [@jmrothberg](https://github.com/jmrothberg)

## License

MIT License
