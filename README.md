# Chess AI — Learn Chess from Move Sequences

A transformer that learns to play chess by reading raw move sequences. Training uses **no chess engine, no search tree, no handcrafted rules** — only next-move prediction on millions of Stockfish games.

The **Pygame client** (`Chess_9_20_26.py`) adds a separate **classical CPU Search** engine. That search does **not** call the neural net. You can play with **Search** on one side and **Neural** (transformer) on the other, or any mix.

## How It Works

The model reads chess games as sequences of UCI moves and learns to predict the next move. Training data is loaded directly from **parquet files** containing Stockfish self-play games — no preprocessing required.

```
<STARTGAME> <W> e2e4 e7e5 g1f3 b8c6 f1b5 ... <EOFG>
```

Two tokenization modes are available, selected at training startup:

### Classic Mode (1 token per move)

Each UCI move is a single token from a ~20K vocabulary. The vocabulary covers all possible moves: 64 origin squares x 63 destination squares x 5 promotion options (none, queen, rook, bishop, knight) = 20,160 move tokens + 5 special tokens.

- **Vocabulary**: ~20K tokens (+ result tokens in result-aware training)
- **Default context** (result-aware trainer): 512 plies (~99.5% of Stockfish games)
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

- **Vocabulary**: 140 tokens (142 with `<B>` / `<U>` in result-aware training)
- **Default context** (result-aware trainer): 1536 tokens = 384 plies (~97% of games)
- **Loss**: Weighted sum across 4 role-specific heads (FROM/TO dominate learning)
- **Generation**: 4 sequential forward passes (one per token role)

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
|-----------------|----------|
| `Chess_Brain_mp_spawn_9_20_26.py` | **Preferred** training (result-aware, packing, value head, multi-GPU / single-GPU) — Sept 20, 2026 |
| `Chess_Brain_mp_spawn_4_12_26.py` | Same trainer, older filename (keep if a run is already using it) |
| `Chess_Brain_3_21_26.py` | Older training script (both token modes; no result-aware features) |
| `Chess_Inference.py` | Neural inference (auto-detects classic vs 4-token from checkpoint) |
| `Chess_9_20_26.py` | Pygame GUI: human / **Search** (CPU) / **Neural** (LLM) per side — Sept 20, 2026 |
| `Chess_4_8_26.py` | Same GUI, older filename (keep if a session is already running it) |
| `Chess_LLM_models/` | Optional local folder for `*.pth` copies (directory tracked; weight files not in git) |
| `plot_loss_March_20_26.py` | Training loss visualization |
| `chess_requirement.txt` | Python dependencies for this repo |
| `readme_mp_spawn.md` | DDP / mp.spawn trainer notes + pitfalls |
| `OLD chess brains/` | Archived older training / data scripts (superseded filenames, kept for reference) |

## Checkpoints

Training writes `.pth` files under folders like `/home/jonathan/Data/Chess_Model_*` (filename encodes layers, heads, embd, batch, epoch, loss, timestamp).

The GUI (**W** / **B**) scans, newest first:

1. `CHESS_LLM_DIR` if set
2. `Chess_LLM_models/` in the repo
3. `/home/jonathan/Data/Chess_Model_*` and `/data/Data/Chess_Model_*`

Git **does not** store weight files. On a machine without the training dumps, copy a `.pth` into `Chess_LLM_models/` or point `CHESS_LLM_DIR` at the folder that has it.

## `OLD chess brains/`

Older copies of dataset/training utilities (e.g. per-game torchcompile brain, parquet converter, combine script) live here for reference. Preferred training is `Chess_Brain_mp_spawn_9_20_26.py` with the current inference/GUI files above.

## Training

**Where to train (Sept 20, 2026):** use a **NVIDIA Linux** machine — **DGX Spark / GB10** (primary) or **multi-GPU Ubuntu**. Do **not** treat the Mac as a training box; Mac is for **playing** with a checkpoint you already trained (see [Play on a Mac](#play-on-a-mac-or-any-machine-without-the-training-setup) below).

### Data
- **Source**: Stockfish self-play games
- **Format**: Parquet files with `Moves` (list of UCI strings) and `Result` columns, or plain text (one game per paragraph)
- **Loading**: Parquet files are read directly in the training script

### Quick Start (NVIDIA Linux only)
```bash
pip install -r chess_requirement.txt
source .venv/bin/activate   # if using the project venv
python Chess_Brain_mp_spawn_9_20_26.py
```

At startup, choose:
1. **New model or load checkpoint** — existing checkpoints auto-detect their mode
2. **Token mode** (new models only) — Classic or 4-Token
3. **Training data** — select a `.parquet` or `.txt` file
4. Architecture / batch / epochs — take the GB10-recommended batch unless you know better

Training supports Ctrl+C to pause and change learning rate or load new data without losing optimizer state. Checkpoints save every 500 batches.

### Hardware / platforms (Sept 20, 2026)

| Machine | Train? | Play GUI? | Notes |
|---------|--------|-----------|--------|
| **DGX Spark / GB10** | **Yes — primary** | Yes | Single-GPU + unified-memory caps. Set `PYTORCH_ALLOC_CONF` (see trainer header). |
| **Multi-GPU Ubuntu (NVIDIA)** | **Yes** | Yes | NCCL DDP via `mp.spawn`. Pick 2+ CUDA GPUs at the prompt. Batch size = total across GPUs. |
| **Apple Mac** | **No (not supported)** | **Yes** | Copy a `.pth` from Spark/Ubuntu. Run `Chess_9_20_26.py`. Optional `CHESS_DEVICE=mps`. Multi-GPU DDP **cannot** run on Mac (NCCL is CUDA-only). Large parquet training on MPS is not a supported workflow. |

**Rule of thumb:** train on Spark or multi-GPU Ubuntu → copy the `.pth` → play on the Mac (or any lighter machine).

Tested on 4x NVIDIA RTX 6000 Ada (48GB each) and NVIDIA GB10 (DGX Spark). Automatic batch size estimation based on actual free GPU memory.

On GB10, `torch.compile` needs CUDA 13's `ptxas` (`/usr/local/cuda/bin/ptxas`). Triton's bundled CUDA 12.8 assembler rejects `sm_121a`. The Chess `.venv` points Triton at the system assembler automatically; or `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.

GB10 shares one RAM pool with the desktop. The trainer keeps **16 GB** of currently free RAM unused and sizes the batch from the rest (typically ~80–100, not 21). Take the recommended batch; do not type 512.

### Result-aware training (`Chess_Brain_mp_spawn_9_20_26.py`, Sept 20, 2026)

The DDP script learns from **who won**, still using only the move sequences plus the `Result` column:

- **Three result tokens.** Games start `<STARTGAME> <W|D|B> moves… <EOFG>`. `<B>` (black won, `0-1`) is new; it used to be lumped into `<D>`. Vocab grows to 142 (4-token) / 20,167 (classic); all existing ids are unchanged.
- **Play like the winner.** The GUI now prompts with `<STARTGAME> <W>` when the neural side is White and `<STARTGAME> <B>` when Black, matching how the training games are written.
- **Winner-weighted loss.** Moves by the side that lost are weighted `loser_move_weight` (0.5); winner and draw moves 1.0.
- **Value head.** A 3-way head predicts W/D/B from the moves so far. To stop it cheating, in `value_mask_prob` (50%) of games the result token is replaced by `<U>` in the input. In the GUI the legal candidates are re-ranked by `log(policy) + VALUE_RERANK_LAMBDA * (P(win) - P(loss))`.
- **Game packing.** Each training row holds as many whole games as fit (position ids restart per game), so far fewer `<PAD>` tokens per batch. Defaults cover real Stockfish length: classic 512 plies (~99.5% of games), 4-token 1536 tokens = 384 plies (~97%).
- **Training display** unpacks one game from the row and prints the input half, the model's predicted moves, the actual moves, and the value head's W/D/B for that position.

Older checkpoints still load and can be resumed: the embedding table is expanded for the new tokens and a value head is added (random init); packing stays off for models trained with absolute positions.

## Playing (GUI)

```bash
python Chess_9_20_26.py
```

Press **H** in-game for help (PLAY / ENGINES / NEURAL / FILES).

| Key | Action |
|-----|--------|
| **a** / **z** | Cycle White / Black between **Search** and **Neural** |
| **W** / **B** | Pick a `.pth` for White / Black Neural (newest first; **Enter** or **1** = latest) |
| **Up** / **Down** | Search depth (default **4**) |
| **x** / **v** | Human vs AI / both sides AI |
| **Left** / **Right** | Step history **ply-by-ply** (restores board, castling rights, en passant) |
| **s** / **l** / **r** | Save / load game / restart |

Click a piece to select it — **legal destinations light up** (green = empty, red ring = capture). Click again to move, or click another own piece to switch.

Neural suggestions are filtered to legal UCI by the GUI. Search uses its own board eval (not the transformer).

### Classical Search engine (Sept 20, 2026)

Built-in Search is alpha-beta with iterative deepening — separate from Neural, and **not** Stockfish:

- PVS, null-move pruning (skipped in late endgame), late-move reductions, check extensions
- Quiescence with MVV-LVA + delta pruning; killers + history heuristic
- Zobrist TT with exact/lower/upper flags, kept across moves (capped size)
- Eval: material + piece-square tables, true passed pawns, bishop pair, open-file / 7th-rank rooks, king safety blended middlegame→endgame
- **Threefold repetition** along the game + search path scores as a draw (0), so a winning Search side avoids shuffling to repeat; a losing side may take the draw

For a world-class opponent instead of the built-in Search:

```bash
CHESS_USE_STOCKFISH=1 python Chess_9_20_26.py
# optional: CHESS_STOCKFISH=/path/to/stockfish
```

Optional: `CHESS_SEARCH_WORKERS=N` parallelizes the root split of Search on Linux (`fork`).

### Play on a Mac (or any machine without the training setup)

**Mac = play only.** Do not run `Chess_Brain_mp_spawn_*.py` for real training on a Mac. Train on DGX Spark or multi-GPU Ubuntu, then bring the checkpoint here.

A checkpoint trained on the DGX (or Ubuntu) plays anywhere; inference needs only PyTorch and Pygame:

```bash
pip install torch pygame
# copy the .pth into Chess_LLM_models/ (or point CHESS_LLM_DIR at your folder)
python Chess_9_20_26.py
```

Inference runs on the CPU by default (`Chess_Inference.py` loads with `map_location="cpu"`). Set `CHESS_DEVICE=mps` to use the Apple GPU, or `CHESS_DEVICE=cuda` on an NVIDIA machine. Everything about the model (token mode, vocab, value head, block size) is read from the checkpoint, so White and Black can use different checkpoints at the same time.

## Author

**Jonathan M. Rothberg** — [@jmrothberg](https://github.com/jmrothberg)

## License

MIT License
