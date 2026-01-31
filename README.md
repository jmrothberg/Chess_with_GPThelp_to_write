# Chess AI Project

A comprehensive chess AI ecosystem with data processing, model training, and interactive gameplay. This project contains two main components: **ChessBrain** (AI model training/learning) and **Chess Game** (interactive chess playing with optional AI opponents).

## 🏗️ Project Components

### 1. ChessBrain (AI Training/Learning)
- **File**: `ChessBrain_Multi_per_game_torchcompile_12_13_25.py`
- **Purpose**: Trains transformer models to predict chess moves from game data
- **Architecture**: Chess-optimized transformer with MultiQueryAttention, RMSNorm, and SwiGLU activations
- **Training Strategy**: "Per-Game" learning - trains on complete games starting from `<STARTGAME>` tokens
- **Hardware**: Optimized for NVIDIA Blackwell GB10 GPU (128GB unified memory)

### 2. Chess Game (Interactive Playing)
- **File**: `Chess_Dec_16_25.py`
- **Purpose**: Interactive chess game with multiple AI opponent options
- **Features**: 2D/3D visualization, legal move highlighting, game history navigation
- **AI Options**: Traditional algorithms (min-max, alpha-beta) + optional LLM integration
- **GUI**: Pygame-based with click-to-move interface

## 📊 Data Processing Pipeline

### Step 1: Download Raw Chess Games
- **Source**: [LAION Strategic Game Chess Dataset](https://huggingface.co/datasets/laion/strategic_game_chess)
- **Format**: Parquet files containing chess games in various formats
- **Location**: `chess_parquet/` directory (excluded from git due to size)

### Step 2: Convert to UCI Format with Promotions
- **Script**: `parquettorext_withpromotion_Dec_12.py`
- **Input**: `chess_parquet/*.parquet` files
- **Output**: `chess_txt_promotion_win/*.txt` files
- **Processing**:
  - Converts moves to UCI format (e2e4, e7e8q for promotions)
  - Preserves pawn promotions (5th character: q=queen, r=rook, b=bishop, n=knight)
  - Adds game result tokens: `<W>` (white win), `<D>` (draw), `<L>` (black win)
  - Filters invalid moves using regex pattern

### Step 3: Combine Datasets for Training
- **Script**: `combine_chess_datasets.py`
- **Input**: `chess_txt_promotion_win/chess_*.txt` files
- **Output**: `combined_chess_datasets/combined_*.txt` files
- **Grouping**: Combines multiple game files into larger training sets (default: 5 files per group)
- **Purpose**: Creates diverse training data with better batch efficiency

## 🧠 ChessBrain Training Process

1. **Data Loading**: Reads combined chess game files
2. **Tokenization**: Converts UCI moves + result tokens into numerical tokens
3. **Per-Game Training**: Each sequence starts with `<STARTGAME>` for proper game boundary masking
4. **Factorized Prediction**: Separate heads for FROM square, TO square, and promotion piece
5. **Value Learning**: Predicts game outcomes using `<W>/<D>/<L>` result tokens

### Training Optimizations
- **MultiQueryAttention**: Efficient attention for long sequences
- **RMSNorm**: Improved training stability vs LayerNorm
- **SwiGLU**: Better activation function than ReLU
- **Game Masking**: Prevents attention across game boundaries
- **Torch Compile**: JIT compilation for faster training

## 🎮 Chess Game Features

### AI Opponent Options
- **Traditional AI**: Min-max algorithm with alpha-beta pruning, quiescence search
- **LLM Integration**: Optional GPT/LLM move suggestions (requires API key)
- **Custom Models**: Play against trained ChessBrain models

### Game Features
- **Complete Chess Rules**: Castling, en passant, pawn promotion, check/checkmate
- **3D Visualization**: Optional 3D board view with piece animations
- **Move Validation**: Highlights legal moves, prevents illegal moves
- **Game History**: Navigate through move history, save/load games
- **Draw Detection**: Threefold repetition, 50-move rule, stalemate

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/jmrothberg/Chess_with_GPThelp_to_write.git
cd Chess_with_GPThelp_to_write
pip install -r chess_requirement.txt
```

### Data Processing (Optional - for training)
```bash
# Convert parquet files to UCI format
python parquettorext_withpromotion_Dec_12.py

# Combine datasets for training
python combine_chess_datasets.py 5 chess_txt_promotion_win/ combined_chess_datasets/
```

### Play Chess
```bash
python Chess_Dec_16_25.py
```

### Train ChessBrain Model
```bash
python ChessBrain_Multi_per_game_torchcompile_12_13_25.py
```

## 📁 Project Structure

```
Chess/
├── ChessBrain_Multi_per_game_torchcompile_12_13_25.py  # Main AI training model
├── Chess_Dec_16_25.py                                  # Interactive chess game
├── Chess_Inference_Dec_14_25.py                       # Model inference utilities
├── parquettorext_withpromotion_Dec_12.py              # Data conversion (parquet → UCI)
├── combine_chess_datasets.py                           # Dataset combination tool
├── game_selector3D.py                                  # 3D game selection UI
├── plot_loss_Nov_9_25.py                               # Training loss visualization
├── chess_requirement.txt                               # Python dependencies
├── ChessBrain_README.md                               # AI training details
├── README_CHESS_PER_GAME.md                           # Per-game training explanation
├── ARIALUNI.TTF                                        # Font for 3D rendering
├── .gitignore                                          # Excludes large datasets
├── chess_parquet/                                      # Raw chess data (excluded)
├── chess_txt_promotion_win/                           # Processed UCI games (excluded)
├── combined_chess_datasets/                           # Combined training data (excluded)
└── Saved Games/                                        # Game save files (excluded)
```

## 🔧 Configuration

### Blackwell GPU Training (Recommended)
Add to your shell profile or `.venv/bin/activate`:
```bash
export PYTORCH_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.8"
export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDA_AUTO_BOOST=0
```

### LLM Integration (Optional)
Set API keys for LLM move suggestions:
```bash
export OPENAI_API_KEY="your-key-here"
```

## 📈 Performance & Hardware

### Supported Hardware
- **NVIDIA Blackwell GB10**: 128GB unified memory (recommended)
- **Apple Silicon Mac Studio**: MPS acceleration
- **Multi-GPU CUDA**: 4x A6000 Ada or similar

### Training Performance
- **Batch Sizes**: 64-1024 (Blackwell-optimized)
- **Sequence Length**: 256 tokens (chess moves + game boundaries)
- **Memory**: 128GB Blackwell handles large models efficiently
- **Speed**: ~100x faster epochs with per-game masking vs random windows

## 🤝 Development

### ChessBrain vs Chess Game
- **ChessBrain**: Focuses on learning/prediction from data (training phase)
- **Chess Game**: Focuses on playing/interaction (inference phase)
- **LLM Option**: Chess game can use LLMs for move suggestions (separate from ChessBrain models)

### Data Flow
```
Raw Games (Parquet) → UCI Conversion → Dataset Combination → ChessBrain Training → Model → Chess Game
     ↓                        ↓                        ↓                        ↓            ↓
  Download               Preserve Promotions        Group Files            Learn Patterns  Play Chess
```

## 📝 Notes

- **Datasets Excluded**: Large chess datasets (>100GB) are not included in this repository
- **Model Files**: Trained ChessBrain models are not included (can be generated via training)
- **Dependencies**: See `chess_requirement.txt` for Python package requirements

## 👨‍💻 Author

**Jonathan M. Rothberg** - [@jmrothberg](https://github.com/jmrothberg)

## 📄 License

MIT License

## 🔗 Related Projects

- [Brain6](https://github.com/jmrothberg/Brain6) - General-purpose transformer training framework
- [LAION Strategic Game Chess](https://huggingface.co/datasets/laion/strategic_game_chess) - Source chess dataset