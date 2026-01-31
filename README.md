# Chess with GPT-4 Help

A Python chess game with AI opponent, written with assistance from GPT-4. Features 3D visualization and the ability to play against custom transformer models trained with [Brain6](https://github.com/jmrothberg/Brain6).

## Features

- **Full Chess Implementation**: Complete chess rules including castling, en passant, and pawn promotion
- **AI Opponent**: Min-max algorithm with alpha-beta pruning
- **3D Visualization**: Optional 3D board view for enhanced gameplay
- **Custom AI Models**: Play against transformer-based models trained on chess games
- **Interactive GUI**: Click-to-move interface with legal move highlighting

## Installation

```bash
git clone https://github.com/jmrothberg/Chess_with_GPThelp_to_write.git
cd Chess_with_GPThelp_to_write
pip install pygame numpy
```

## Usage

```bash
python chess_game.py
```

## Development Story

I always wanted to write a chess game that could beat me. With GPT-4 as my coding assistant, I finally did it.

**What worked well:**
- GPT-4 was excellent at finding bugs
- Generated good function outlines and structure
- Helped with boilerplate code

**What required manual work:**
- Game logic, especially min-max algorithm
- GPT-4 would write good outlines but get the logic wrong
- Had to debug and fix algorithm details by hand

**Lesson learned:** AI coding assistants are great collaborators, but complex logic still requires human debugging.

## Related Projects

- [Brain6](https://github.com/jmrothberg/Brain6) - Train your own transformer models on chess games

## Author

**Jonathan M. Rothberg** - [@jmrothberg](https://github.com/jmrothberg)

## License

MIT License
