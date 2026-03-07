#JMR Simple Chess Game March 4 2024
#Removed large number of the AI methods and board evaluations to keep code base smaller for LLM help
#Added LLM to game Sept 19
#Added simple notation for internal use and use of chess move tokenizer Sept 24, promotes to knight if checkmate
#Added top k responses from LLM to game and 3 and 5 move repetition for draw   Sept 25
#Changed to CPU Sept 26 and also made sure no crash when no moves are found but does stalemate.
#Allow board to be rotated 180 degrees. 3D board added Sept 27
#Cleaned up save game Sept 29, fallback is best evaluation
#Draw logic corrected Sept 30
#Added ability to have different AI for each color Oct 5
#Added quiescence search Oct 9 for best_improved
#Oct 15, bug was clearing the screen for 2D board, and need to just do for 3D board
#October 23 simplified game history and position history so you can go back and forth without errors
# November 5, added mobile LLM support
# October 12 simplified program just two options
# October 13 2025 - Major AI thinking optimizations: persistent transposition table for 30-60% speedup,
# killer moves heuristic for better move ordering, 
# and optimized quiescence search with MVV-LVA capture ordering
import sys
import pygame
import copy
import time
import os
import string
import json
import Chess_Inference_Dec_14_25_WB_2_12_26 as brain_inference
from Chess_Inference_Dec_14_25_WB_2_12_26 import generate_response
from game_selector3D import loop_to_select_new_game, draw_pieces_not_on_board
import platform
try:
    from numba import jit, int32, float32
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available, using Python evaluation")

# Initialize Pygame
pygame.init()

# Constants for the game
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 1400
BOARD_SIZE = 8
SQUARE_SIZE = SCREEN_WIDTH // BOARD_SIZE
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
OFF_WHITE = (200, 200, 200)  # Slightly darker than pure white

sound_enabled = True
board_reversed = False
display_mode = '2D'

# Global AI and LLM variables
ai_method_white = "Improved"
ai_method_black = "Improved"
llm_white = None
llm_black = None
block_size = None

# LLM move statistics tracking
llm_stats = {
    'total_moves': 0,
    'first_legal': 0,
    'second_legal': 0,
    'third_plus_legal': 0,
    'no_legal': 0
}

# Initialize the screen with given dimensions
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
pygame.display.set_caption("JMR's Game of Chess Game: Press H for Help")
font = pygame.font.SysFont("Arial", 42)
font_moves = pygame.font.SysFont("Arial", 33)
font_info = pygame.font.SysFont("Arial", 24)
font_big = pygame.font.Font("ARIALUNI.TTF", 87)
font_2big = pygame.font.Font("ARIALUNI.TTF", 120)

# Chess board representation:
# - 8x8 grid stored as list of lists (board[row][col])
# - Row 0 = Black's back rank, Row 7 = White's back rank
# - Col 0 = a-file (queenside), Col 7 = h-file (kingside)
# - Empty squares = "" (empty string)
# - Pieces = "Color+Type+Number" format:
#   * First char: 'W' = White, 'B' = Black
#   * Second char: 'P' = Pawn, 'N' = Knight, 'B' = Bishop, 'R' = Rook, 'Q' = Queen, 'K' = King
#   * Optional number: Distinguishes rooks (R1=left, R2=right) for castling logic
board = [
        ["BR1", "BN", "BB", "BQ", "BK", "BB", "BN", "BR2"],
        ["BP"] * 8,
        [""] * 8,
        [""] * 8,
        [""] * 8,
        [""] * 8,
        ["WP"] * 8,
        ["WR1", "WN", "WB", "WQ", "WK", "WB", "WN", "WR2"]
]

initial_board = copy.deepcopy(board)

piece_values = {
        'WP': 1, 'WN': 3, 'WB': 3, 'WR1': 5, 'WR2': 5, 'WQ': 9, 'WK': 100,
        'BP': -1, 'BN': -3, 'BB': -3, 'BR1': -5, 'BR2': -5, 'BQ': -9, 'BK': -100
    }

# Positional values will be precomputed after positional_values is defined

pieces_uni = {
    "BR1": u'\u265C',"BR2": u'\u265C', "BN": u'\u265E', "BB": u'\u265D', "BQ": u'\u265B', "BK": u'\u265A', "BP": u'\u265F',
    "WP": u'\u2659', "WR1": u'\u2656',"WR2": u'\u2656', "WN": u'\u2658', "WB": u'\u2657', "WQ": u'\u2655', "WK": u'\u2654'
}

piece_dict = {"BR1": "Black Rook","BR2": "Black Rook", "BN": "Black Knight", "BB": "Black Bishop", "BQ": "Black Queen", "BK": "Black King", "BP": "Black Pawn",
              "WR1": "White Rook", "WR2": "White Rook","WN": "White Knight", "WB": "White Bishop", "WQ": "White Queen", "WK": "White King", "WP": "White Pawn", "W": "White", "B": "Black"}
  
positional_values = {
    "P": [[0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [0.5, 0.5, 1, 1.5, 1.5, 1, 0.5, 0.5],
          [0.25, 0.25, 0.5, 1.25, 1.25, 0.5, 0.25, 0.25],
          [0, 0, 0, 1, 1, 0, 0, 0],
          [0.25, -0.25, -0.5, 0, 0, -0.5, -0.25, 0.25],
          [0.25, 0.5, 0.5, -1, -1, 0.5, 0.5, 0.25],
          [0, 0, 0, 0, 0, 0, 0, 0]],

    "N": [[-1, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -1],
          [-0.5, 0, 0, 0.25, 0.25, 0, 0, -0.5],
          [-0.5, 0.25, 0.5, 0.75, 0.75, 0.5, 0.25, -0.5],
          [-0.5, 0, 0.5, 0.75, 0.75, 0.5, 0, -0.5],
          [-0.5, 0.25, 0.5, 0.75, 0.75, 0.5, 0.25, -0.5],
          [-0.5, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5],
          [-0.5, 0, 0.25, 0.25, 0.25, 0.25, 0, -0.5],
          [-1, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -1]],

    "B": [[-1, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -1],
          [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
          [-0.5, 0, 0.25, 0.5, 0.5, 0.25, 0, -0.5],
          [-0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, -0.5],
          [-0.5, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5],
          [-0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5],
          [-0.5, 0.25, 0, 0, 0, 0, 0.25, -0.5],
          [-1, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -1]],

    "R": [[0, 0, 0, 0, 0, 0, 0, 0],
          [0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25],
          [-0.25, 0, 0, 0, 0, 0, 0, -0.25],
          [-0.25, 0, 0, 0, 0, 0, 0, -0.25],
          [-0.25, 0, 0, 0, 0, 0, 0, -0.25],
          [-0.25, 0, 0, 0, 0, 0, 0, -0.25],
          [-0.25, 0, 0, 0, 0, 0, 0, -0.25],
          [0, 0, 0, 0.25, 0.25, 0, 0, 0]],

    "Q": [[-1, -0.5, -0.5, -0.25, -0.25, -0.5, -0.5, -1],
          [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
          [-0.5, 0, 0.25, 0.25, 0.25, 0.25, 0, -0.5],
          [-0.25, 0, 0.25, 0.25, 0.25, 0.25, 0, -0.25],
          [0, 0, 0.25, 0.25, 0.25, 0.25, 0, -0.25],
          [-0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0, -0.5],
          [-0.5, 0, 0.25, 0, 0, 0, 0, -0.5],
          [-1, -0.5, -0.5, -0.25, -0.25, -0.5, -0.5, -1]],

    "K": [[-1.5, -2, -2, -2.5, -2.5, -2, -2, -1.5],
          [-1.5, -2, -2, -2.5, -2.5, -2, -2, -1.5],
          [-1.5, -2, -2, -2.5, -2.5, -2, -2, -1.5],
          [-1.5, -2, -2, -2.5, -2.5, -2, -2, -1.5],
          [-1, -1.5, -1.5, -2, -2, -1.5, -1.5, -1],
          [-0.5, -1, -1, -1, -1, -1, -1, -0.5],
          [1, 1, 0, 0, 0, 0, 1, 1],
          [1, 1.5, 0.5, 0, 0, 0.5, 1.5, 1]]
}

# Precompute positional values for faster evaluation
get_pos_val_white = {(piece, i, j): value for piece in positional_values.keys()
                     for i, row in enumerate(positional_values[piece])
                     for j, value in enumerate(row)}
get_pos_val_black = {(piece, 7-i, j): -value for piece in positional_values.keys()
                     for i, row in enumerate(positional_values[piece])
                     for j, value in enumerate(row)}

# Precompute piece values as array for Numba
piece_value_array = [0] * 256  # ASCII range
for piece, value in piece_values.items():
    if piece:
        if len(piece) > 1:
            idx = ord(piece[0]) * 16 + ord(piece[1])
        else:
            idx = ord(piece[0])
        if idx < 256:
            piece_value_array[idx] = value

# Convert positional values to 3D array for Numba [piece_type][i][j]
import numpy as np
piece_types = list(positional_values.keys())
pos_val_array = np.zeros((len(piece_types), 8, 8), dtype=np.float32)
for p_idx, piece in enumerate(piece_types):
    for i in range(8):
        for j in range(8):
            pos_val_array[p_idx, i, j] = positional_values[piece][i][j]

# Piece type to index mapping
piece_type_to_idx = {piece: idx for idx, piece in enumerate(piece_types)}

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def evaluate_board_numba(board_flat, pos_val_array):
        """
        Numba-compiled board evaluation using precomputed arrays.
        board_flat: flattened board representation
        pos_val_array: 3D array [piece_type][i][j] of positional values
        """
        evaluation = 0.0

        for idx in range(64):
            i = idx // 8
            j = idx % 8
            piece_code = board_flat[idx]

            if piece_code != 0:  # Not empty
                # Extract piece type from code (simplified mapping)
                if piece_code < 200:  # White pieces
                    piece_type_idx = (piece_code - 87) // 2  # Map ASCII to piece type index
                    is_white = True
                else:  # Black pieces
                    piece_type_idx = (piece_code - 119) // 2  # Map ASCII to piece type index
                    is_white = False

                # Get piece value (simplified)
                piece_val = 0
                if piece_type_idx == 0: piece_val = 4  # Rook
                elif piece_type_idx == 1: piece_val = 3  # Knight/Bishop
                elif piece_type_idx == 2: piece_val = 3  # Knight/Bishop
                elif piece_type_idx == 3: piece_val = 9  # Queen
                elif piece_type_idx == 4: piece_val = 1  # Pawn
                elif piece_type_idx == 5: piece_val = 100  # King

                if not is_white:
                    piece_val = -piece_val

                # Get positional value
                if piece_type_idx < pos_val_array.shape[0]:
                    if is_white:
                        pos_val = pos_val_array[piece_type_idx, i, j]
                    else:
                        pos_val = -pos_val_array[piece_type_idx, 7-i, j]
                else:
                    pos_val = 0

                evaluation += piece_val + pos_val

        return evaluation
else:
    def evaluate_board_numba(board_flat, pos_val_array):
        return 0  # Fallback

# Constants for isometric projection
TILE_WIDTH = 128   # Width of each tile in pixels
TILE_HEIGHT = 64  # Height of each tile in pixels
BOARD_ORIGIN = (SCREEN_WIDTH // 2, 200)  # Origin point for drawing the board


# Add these wrapper functions
def draw_board_wrapper(screen, board):
    
    if display_mode == '2D':
        draw_board(screen)
        draw_pieces(screen, board)
    else:
        screen.fill(WHITE)
        draw_isometric_board(screen)
        draw_isometric_pieces(screen, board)

def screen_to_board_position(mouse_pos):
    if display_mode == '2D':
        return screen_to_board_pos(mouse_pos)
    else:
        return screen_to_board_pos_isometric(mouse_pos)

# Functions for isometric projection
def board_to_iso(x, y):
    """Converts board coordinates to isometric screen coordinates."""
    iso_x = (x - y) * (TILE_WIDTH // 2) + BOARD_ORIGIN[0]
    iso_y = (x + y) * (TILE_HEIGHT // 2) + BOARD_ORIGIN[1]
    return iso_x, iso_y

def screen_to_board_pos_isometric(mouse_pos):
    """Converts screen coordinates to board positions on the isometric board, accounting for board reversal."""
    x, y = mouse_pos
    x -= BOARD_ORIGIN[0]
    y -= BOARD_ORIGIN[1]

    # Calculate the approximate board coordinates
    float_col = (x / (TILE_WIDTH / 2) + y / (TILE_HEIGHT / 2)) / 2
    float_row = (y / (TILE_HEIGHT / 2) - (x / (TILE_WIDTH / 2))) / 2

    col = int(float_col)
    row = int(float_row)

    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        # Adjust for board reversal
        board_row = BOARD_SIZE - 1 - row if board_reversed else row
        board_col = BOARD_SIZE - 1 - col if board_reversed else col
        return board_row, board_col
    else:
        return None

def draw_isometric_board(screen):
    """Draws the chessboard using isometric projection, accounting for board reversal."""
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            # Adjust row and col based on board reversal
            display_row = BOARD_SIZE - 1 - row if board_reversed else row
            display_col = BOARD_SIZE - 1 - col if board_reversed else col

            # Get the coordinates for the four corners of the tile
            top = board_to_iso(display_col, display_row)
            right = board_to_iso(display_col + 1, display_row)
            bottom = board_to_iso(display_col + 1, display_row + 1)
            left = board_to_iso(display_col, display_row + 1)
            tile = [top, right, bottom, left]

            # Choose color based on the square
            color = BLUE if (display_row + display_col) % 2 == 0 else GRAY

            # Draw the tile
            pygame.draw.polygon(screen, color, tile)
            # Optionally, draw outlines
            pygame.draw.polygon(screen, BLACK, tile, 1)

def draw_isometric_pieces(screen, board):
    """Draws the chess pieces on the isometric board, accounting for board reversal."""
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            # Adjust row and col based on board reversal
            board_row = BOARD_SIZE - 1 - row if board_reversed else row
            board_col = BOARD_SIZE - 1 - col if board_reversed else col

            piece = board[board_row][board_col]
            if piece:
                # Calculate the display positions based on board reversal
                display_row = BOARD_SIZE - 1 - board_row if board_reversed else board_row
                display_col = BOARD_SIZE - 1 - board_col if board_reversed else board_col

                # Calculate the center position for the piece
                x, y = board_to_iso(display_col + 0.5, display_row + 0.5)
                y -= TILE_HEIGHT // 2  # Adjust y to position piece correctly

                # Optionally adjust y to better center the piece vertically
                y_offset = 10  # Adjust this value as needed
                y -= y_offset

                # Draw the piece symbol
                if piece[0] == "W":
                    text = font_big.render(pieces_uni[piece], True, OFF_WHITE)
                else:
                    text = font_big.render(pieces_uni[piece], True, BLACK)
                text_rect = text.get_rect(center=(x, y))
                screen.blit(text, text_rect)

# Helper function to convert mouse position to board coordinates
def screen_to_board_pos(mouse_pos):
    x, y = mouse_pos
    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE
    if board_reversed:
        row = BOARD_SIZE - 1 - row
        col = BOARD_SIZE - 1 - col  
    return row, col


def save_game(board, move_number, player, ai, depth, evaluation_method, ai_method_white, ai_method_black, depth_equation, show_simulation, list_of_boards, position_history, has_moved_history, game_history, game_history_simple):
    os.makedirs("Saved Games", exist_ok=True)
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Saved game as board_{datetime}.json")
    with open(f"Saved Games/board_{datetime}.json", "w") as f:
        json.dump({
            "board": board,
            "move_number": move_number,
            "player": player,
            "ai": ai,
            "depth": depth,
            "evaluation_method": evaluation_method,
            "ai_method_white": ai_method_white,
            "ai_method_black": ai_method_black,
            "depth_equation": depth_equation,
            "show_simulation": show_simulation,
            "list_of_boards": list_of_boards,
            "position_history": position_history,
            "has_moved_history": has_moved_history,
            "game_history": game_history,
            "game_history_simple": game_history_simple
        }, f)


# Does all moves BUT not promotions! so if only use one function would need to do promotions
def get_moves_for_piece(board, start_row, start_col, last_move = None, check_castling = True):
    #Never changes board
    directions = {
        'P': [(-1, 0)], # White pawns move up; will need to multiply by color direction for black pawns
        'N': [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)],
        'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)], # Bishops, rooks and queens can move further in one turn
        'R': [(-1, 0), (0, -1), (0, 1), (1, 0)],   # We'll expand these in the next function
        'Q': [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
        'K': [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    }

    piece = board[start_row][start_col]
    piece_type = piece[1]
    color = piece[0]  # This is either "W" or "B"
    moves = []

    for direction in directions[piece_type]:
        # Determine how many steps in a direction the piece is allowed to go
        steps = 1 if piece_type in "NKP" else BOARD_SIZE
        for step in range(1, steps + 1):
            target_row = start_row + direction[0] * step
            target_col = start_col + direction[1] * step
            # Check bounds
            if 0 <= target_row < BOARD_SIZE and 0 <= target_col < BOARD_SIZE:
                target_piece = board[target_row][target_col]
                # Check target square - empty or containing an opponent's piece
                if target_piece == "" or target_piece[0] != color:
                    moves.append(((start_row, start_col), (target_row, target_col)))
                # Stop at the first piece in a direction
                if target_piece:
                    break
            else:
                break
     # Update for pawn moves: handle initial move and capturing diagonally
    if piece_type == 'P':
        direction = -1 if color == "W" else 1
        moves = []  # Reset moves for pawns, since their mechanics are unique
        # Move forward
        next_row = start_row + direction
        if 0 <= next_row < BOARD_SIZE and board[next_row][start_col] == "":
            moves.append(((start_row, start_col), (next_row, start_col)))
            # Initial double move for pawns
            if (color == "W" and start_row == 6) or (color == "B" and start_row == 1):
                next_next_row = start_row + 2 * direction
                if 0 <= next_next_row < BOARD_SIZE and board[next_next_row][start_col] == "":
                    moves.append(((start_row, start_col), (next_next_row, start_col)))
        # Diagonal captures
        for offset in (-1, 1):
            diag_row = start_row + direction
            diag_col = start_col + offset
            if 0 <= diag_row < BOARD_SIZE and 0 <= diag_col < BOARD_SIZE:
                target_piece = board[diag_row][diag_col]
                if target_piece and target_piece[0] != color:
                    moves.append(((start_row, start_col), (diag_row, diag_col)))

        # ======================================================================================
        # EN PASSANT CAPTURE RULE
        # ======================================================================================
        #
        # Special pawn capture of enemy pawn that just moved two squares forward.
        #
        # HOW IT WORKS:
        # 1. Opponent pawn moves from 2nd to 4th rank (or 7th to 5th for black)
        # 2. Your pawn must be on adjacent file and 4th rank (white) or 5th rank (black)
        # 3. You can capture "en passant" on next move only, as if pawn only moved one square
        # 4. Capturing pawn moves diagonally to empty square behind enemy pawn
        # 5. Enemy pawn is removed from board (not where capturing pawn lands)
        #
        # EXAMPLE: e2-e4, then d7-d5, then e4xd5 en passant (dxe5 in notation)
        #
        # ======================================================================================
        # Put en passant capture logic here to make sure it is added to moves
        if (color == "W" and start_row == 3) or (color == "B" and start_row == 4):
            if last_move is not None:
                last_start_pos, last_end_pos = last_move
                last_piece = board[last_end_pos[0]][last_end_pos[1]]
                # Ensure the last move was a pawn moving two steps forward to the adjacent column
                if last_piece and last_piece[1] == 'P' and abs(last_start_pos[0] - last_end_pos[0]) == 2 and abs(last_end_pos[1] - start_col) == 1:
                    en_passant_capture_row = last_end_pos[0]  # Row where the last moved pawn landed
                    en_passant_capture_col = last_end_pos[1]  # Column of the last moved pawn
                    en_passant_vulnerable_row = 3 if color == 'W' else 4
                    # Ensure the capturing pawn is on the correct row to perform en passant
                    # capture is one row less than the row where the last moved pawn landed
                    if start_row == en_passant_vulnerable_row:
                        moves.append(((start_row, start_col), (en_passant_capture_row + direction, en_passant_capture_col)))

    # ======================================================================================
    # CASTLING RULE - KING SAFETY MOVE
    # ======================================================================================
    #
    # King moves two squares toward rook, rook jumps to other side of king.
    #
    # KING-SIDE CASTLING (O-O):
    # - King from e1 to g1 (white) or e8 to g8 (black)
    # - Rook from h1 to f1 (white) or h8 to f8 (black)
    #
    # QUEEN-SIDE CASTLING (O-O-O):
    # - King from e1 to c1 (white) or e8 to c8 (black)
    # - Rook from a1 to d1 (white) or a8 to d8 (black)
    #
    # REQUIREMENTS:
    # 1. King and rook haven't moved yet
    # 2. No pieces between king and rook
    # 3. King not in check
    # 4. King doesn't pass through check
    # 5. King doesn't end in check
    #
    # ======================================================================================
    # Add castling for the king
    if piece_type == 'K' and check_castling:
        if not has_moved[piece]:
            if (color == 'W' and start_row == 7 and start_col == 4) or (color == 'B' and start_row == 0 and start_col == 4):
                # Check if the rook on the side that is moving has moved
                opponent_color = 'B' if color == 'W' else 'W'
                if board[start_row][0] == color + 'R1' and not has_moved[color + 'R1']:
                    if all(board[start_row][i] == '' for i in range(1, 4)):
                        if not any(is_square_under_attack(board, start_row, i, opponent_color) for i in range(start_col, 2)):
                            moves.append(((start_row, start_col), (start_row, 2)))
                if board[start_row][7] == color + 'R2' and not has_moved[color + 'R2']:
                    if all(board[start_row][i] == '' for i in range(5, 7)):
                        if not any(is_square_under_attack(board, start_row, i, opponent_color) for i in range(start_col, 6)):
                            moves.append(((start_row, start_col), (start_row, 6)))
    return moves  
           

def get_all_legal_moves(board, color, last_move=None, check_legality=True):
    move_candidates = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece.startswith(color):
                move_candidates.extend(get_moves_for_piece(board, row, col, last_move=last_move))

    if not check_legality:
        return move_candidates

    # do the legality checks after you've gathered all the candidates
    legal_moves = [move for move in move_candidates if is_move_legal(board, move, color)]
    return legal_moves

def is_in_check(board, color):
    # Find the king's position once and for all
    king_position = next(((r, c) for r in range(BOARD_SIZE)
                          for c in range(BOARD_SIZE)
                          if board[r][c] == f"{color}K"), None)
    # Get all moves for the opponent
    opponent_color = "W" if color == "B" else "B"
    opponent_moves = get_all_legal_moves(board, opponent_color, check_legality=False)
    # See if any move attacks the king's position
    return any(end == king_position for _, end in opponent_moves)


def is_square_under_attack(board, row, col, attacker_color):
    """REVERTED TO WORKING ORIGINAL - generates moves for all enemy pieces and checks if any attack the square."""
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = board[r][c]
            if piece.startswith(attacker_color):
                moves = get_moves_for_piece(board, r, c, last_move=None, check_castling=False)
                if any(end == (row, col) for _, end in moves):
                    return True
def is_move_legal(board, move, color):
    #Would change board
    legal_board = copy.deepcopy(board) # since not using simulation that copies the board
    start, end = move
    piece = legal_board[start[0]][start[1]]
    legal_board[end[0]][end[1]] = piece
    legal_board[start[0]][start[1]] = ""
    if is_in_check(legal_board, color):
        return False
    return True
def is_checkmate(board, color):
    if not get_all_legal_moves(board, color) and is_in_check(board, color):
        return True
def is_stalemate(board, color):
    return not get_all_legal_moves(board, color) and not is_in_check(board, color)

def board_to_hashable(board, player_to_move): #need to make board hashable for transposition table for stalemate
    return (tuple(tuple(row) for row in board), player_to_move)

# ======================================================================================
# POSITION REPETITION DETECTION FOR DRAWS
# ======================================================================================
#
# Chess rule: Threefold repetition = draw (player can claim)
# Fivefold repetition = automatic draw
#
# HOW IT WORKS:
# 1. Tracks all board positions in position_history (after each move)
# 2. Converts board to hashable tuple for comparison
# 3. Counts how many times current position appeared before
# 4. Threefold: Player can claim draw (strategic decision)
# 5. Fivefold: Automatic draw (prevents infinite loops)
#
# WHY IMPORTANT:
# - Prevents engines from forcing repetition to avoid loss
# - AI evaluates whether to claim draw based on material disadvantage
#
# ======================================================================================
def is_repetition(count=3):
    if len(position_history) < count * 2 - 1:  # Minimum moves needed for repetition
        return False
    current_position = position_history[-1]
    return position_history.count(current_position) >= count

def can_claim_draw():
    return is_repetition(3)

def is_automatic_draw():
    return is_repetition(5)

def ai_should_claim_draw(board, ai_color):
    # Evaluate the board from White's perspective
    evaluation = evaluate_board(board)

    # Define a threshold for what's considered a losing position
    losing_threshold = -1  # You can adjust this value

    # AI claims draw if it's in a losing position
    if ai_color == 'W':
        return evaluation <= losing_threshold
    else:  # ai_color == 'B'
        return evaluation >= -losing_threshold

def simulate_move(board, move, real_board=False):
    start, end = move
    piece = board[start[0]][start[1]]

    # Safety check: ensure we have a valid piece
    if not piece:
        print(f"ERROR: No piece at {start} for move {move}")
        return board  # Return unchanged board

    captured_piece = board[end[0]][end[1]]

    # Make a shallow copy of the board
    new_board = [row[:] for row in board]

    if real_board or show_simulation:
        notation = convert_to_chess_notation(new_board, move)

    if real_board:
        pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,50))
        pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))
    
    # ======================================================================================
    # PAWN PROMOTION RULE
    # ======================================================================================
    #
    # Pawn reaching opponent's back rank promotes to another piece.
    #
    # NORMAL PROMOTION: Always promotes to Queen (most powerful piece)
    #
    # SPECIAL CASE - KNIGHT PROMOTION FOR CHECKMATE:
    # 1. Test if promoting to Queen leads to checkmate
    # 2. If not, promote to Queen as default
    # 3. If yes, promote to Knight instead (can still checkmate)
    #
    # WHY KNIGHT?: Sometimes Queen promotion blocks checkmate path,
    # but Knight promotion allows the checkmate to succeed.
    #
    # ======================================================================================
    if piece[1] == 'P': #Pawn promotion to queen or knight if checkmate
        if (piece[0] == 'W' and end[0] == 0) or (piece[0] == 'B' and end[0] == BOARD_SIZE - 1):
            original_piece = piece
            
            # Create a temporary board to test knight promotion
            temp_board = copy.deepcopy(new_board)
            temp_board[end[0]][end[1]] = piece[0] + 'N'  # Promote to knight
            
            # Check if promoting to knight leads to checkmate
            opponent_color = 'B' if piece[0] == 'W' else 'W'
            if is_checkmate(temp_board, opponent_color):
                piece = piece[0] + 'N'  # Promote to knight
            else:
                piece = piece[0] + 'Q'  # Promote to queen (default)
            
            new_board[end[0]][end[1]] = piece  # Set the promoted piece
            
            if show_simulation and real_board:
                print(f"Simulated {notation}. {original_piece} promoted to {piece_dict[piece]}")
                 # Comment explaining the promotion decision
                if piece[1] == 'N':
                    print("Promoted to knight as it leads to checkmate")
                else:
                    print("Promoted to queen as default (knight promotion doesn't lead to immediate checkmate)")
            if real_board:
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 195, SCREEN_WIDTH - 50,50))
                screen.blit(font.render(f"{original_piece} promoted to {piece_dict[piece]}", True, BLACK), (27, SCREEN_HEIGHT - 195))
                read_aloud(f"{original_piece} promoted to {piece_dict[piece]}")
            
    # Handle en passant
    if piece[1] == 'P' and abs(start[1] - end[1]) == 1 and new_board[end[0]][end[1]] == "":
        if piece[0] == "W":
            new_board[end[0]+1][end[1]] = ""
            if show_simulation and real_board:
                print(f"Simulated en passant capture. {notation}.")
            if real_board:
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 150, SCREEN_WIDTH - 50,50))
                screen.blit(font.render(f"en passant capture. {notation}", True, BLACK), (27 ,SCREEN_HEIGHT - 150))
                read_aloud(f"en passant capture. {notation}")
        else:
            new_board[end[0]-1][end[1]] = ""
            if show_simulation and real_board:
                print(f"Simulated en passant capture. {notation}.")
            if real_board:
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 150, SCREEN_WIDTH - 50,50))
                screen.blit(font.render(f"en passant capture. {notation}", True, BLACK), (27 ,SCREEN_HEIGHT - 150))
                read_aloud(f"en passant capture. {notation}")
            
    # Handle castling
    if piece[1] == 'K' and abs(start[1] - end[1]) == 2:
        color = piece[0]
        if end[1] == start[1]+2:  # King-side castling
            new_board[start[0]][start[1]+1] = color + 'R2'
            new_board[start[0]][start[1]+3] = ''
            if show_simulation and real_board:
                print(f"Simulated {piece_dict[color]} castling.")
            if real_board:
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 195, SCREEN_WIDTH - 50,50))
                screen.blit(font_info.render(f"{piece_dict[color]} king-side castling.", True, BLACK), (27, SCREEN_HEIGHT - 195))
                read_aloud(f"{piece_dict[color]} king-side castling.")
        elif end[1] == start[1]-2:  # Queen-side castling
            new_board[start[0]][start[1]-1] = color + 'R1'
            new_board[start[0]][start[1]-4] = ''
            if show_simulation and real_board:
                print(f"Simulated {piece_dict[color]} castling.")   
            if real_board:
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 195, SCREEN_WIDTH - 50, 50))
                screen.blit(font_info.render(f"{piece_dict[color]} queen-side castling.", True, BLACK), (27, SCREEN_HEIGHT - 195))
                read_aloud(f"{piece_dict[color]} queen-side castling.")

    # Capture logic 
    if captured_piece != "":
        if show_simulation and real_board:
            if captured_piece[1] == 'K':
                print(f"Simulated Checkmate of {captured_piece}")
            else:
                print(f"Simulated {piece} captures {captured_piece} at {end}")
        if real_board:
            pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 195, SCREEN_WIDTH - 50,50))
            screen.blit(font.render(f"{piece_dict[piece]} captures {piece_dict [captured_piece]}", True, BLACK), (27, SCREEN_HEIGHT - 195))
            read_aloud(f"{piece_dict[piece]} captures {piece_dict [captured_piece]}")
   
    # Perform the move
    new_board[end[0]][end[1]] = piece
    new_board[start[0]][start[1]] = ""       

    if show_simulation:
        draw_board_wrapper(screen, new_board)
        pygame.display.flip()

    return new_board


# ======================================================================================
# MINIMAX WITH ALPHA-BETA PRUNING ALGORITHM
# ======================================================================================
#
# This is the core recursive search function that evaluates chess positions.
#
# HOW IT WORKS:
# 1. Uses minimax algorithm: maximize for AI's turn, minimize for opponent's turn
# 2. Alpha-beta pruning: 'alpha' tracks best score for maximizing player (AI),
#    'beta' tracks best score for minimizing player (opponent)
# 3. When alpha >= beta, the current branch is pruned (won't affect final decision)
# 4. Returns both evaluation score and best move sequence for current depth
#
# KEY OPTIMIZATIONS:
# - Transposition table lookup (30-60% speedup)
# - Killer move heuristic for better move ordering
# - Null move pruning (test if position is so good opponent can pass)
# - Late move reduction (search less deeply for later moves)
# - Quiescence search at leaf nodes for tactical stability
#

# ======================================================================================
def select_best_ai_move_improved(board, depth, color, AI_color, alpha=float('-inf'), beta=float('inf'), display_simulation=False, last_move=None, initial_depth=None):
    global transposition_table, piece_values, depth_formula, discount

    # Use optimizations
    use_transposition = True
    use_killer_moves = True

    # Transposition table lookup
    board_key = (tuple(map(tuple, board)), color, AI_color)  # Include whose turn it is and which AI is playing
    if use_transposition and board_key in transposition_table:
        eval_board, eval_depth, best_move_new = transposition_table[board_key]
        if display_simulation:
           print(f"Transposition Table hit for {color} {depth} {eval_board:.2f} {eval_depth} {best_move_new} {last_move}")
        if eval_depth >= depth:
            return eval_board, best_move_new

    legal_moves = get_all_legal_moves(board, color, last_move=last_move, check_legality=True)
    
    # DEBUG: Check if legal_moves is unexpectedly empty at top level
    if depth == initial_depth:
        print(f"DEBUG ENTRY: color={color}, depth={depth}, initial={initial_depth}, legal_moves={len(legal_moves)}")
    
    check = is_in_check(board, color)

    evaluation = evaluate_board(board)

    if not check and not legal_moves:
        evaluation = evaluation * (discount**((initial_depth + 1) - depth))
        if display_simulation:
            print(f"Stalemate, no legal moves for {color}. Evaluation: {-evaluation:.2f}")
            print(f"Depth: {depth}, Last move: {last_move}")
        return -evaluation, []

    if check and not legal_moves:
        if color == "W":
            evaluation = (evaluation - 100) * (discount**((initial_depth + 1) - depth))
        else:
            evaluation = (evaluation + 100) * (discount**((initial_depth + 1) - depth))
        if display_simulation:
            print(f"Checkmate of {color}. Evaluation: {evaluation:.2f}")
            print(f"Depth: {depth}, Last move: {last_move} of {'W' if color == 'B' else 'B'}")
        return evaluation, []

    if depth <= 0:
        return quiescence_search(board, color, AI_color, alpha, beta, 0, 2)

    # ======================================================================================
    # NULL MOVE PRUNING OPTIMIZATION
    # ======================================================================================
    #
    # Allows opponent to "pass" their turn to test position strength.
    #
    # HOW IT WORKS:
    # 1. Temporarily give opponent an extra move (null move)
    # 2. If opponent still can't create a threat, position must be very strong
    # 3. Reduces search tree by pruning weak positions early
    # 4. Only used when not in check (passing when in check is illegal)
    # 5. Uses reduced depth (depth-3) for efficiency
    #
    # EXAMPLE: If current position is so good that opponent gains nothing from extra move,
    # then we can prune this branch and avoid deeper search.
    #
    # ======================================================================================
    # Null Move Pruning (not at root level - must return actual move at root)
    if depth > 2 and not check and depth < initial_depth:
        null_move_eval, _ = select_best_ai_move_improved(board, depth - 3, 'B' if color == 'W' else 'W', AI_color, -beta, -beta + 1, display_simulation, last_move, initial_depth)
        null_move_eval = -null_move_eval
        if null_move_eval >= beta:
            return beta, []

    # ======================================================================================
    # MOVE ORDERING HEURISTIC - CRITICAL FOR ALPHA-BETA EFFICIENCY
    # ======================================================================================
    #
    # Orders moves to improve alpha-beta pruning effectiveness.
    # Better ordered moves = more pruning = faster search.
    #
    # ORDERING CRITERIA (highest priority first):
    # 1. Killer Moves: Moves that caused beta cutoffs at same depth in other branches
    # 2. Captures: MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    #    - Queen takes Pawn (high priority - saves queen)
    #    - Pawn takes Queen (high priority - gains queen)
    # 3. Center control and advancement bonuses
    # 4. Pawn promotions (very high priority)
    #
    # Killer moves work because: if a move caused cutoff elsewhere, likely good here too
    #
    # ======================================================================================
    # Move Ordering with killer moves heuristic
    def order_moves(board, moves, color):
        global killer_moves
        ordered_moves = []
        for move in moves:
            score = 0
            piece = board[move[0][0]][move[0][1]]
            target = board[move[1][0]][move[1][1]]

            # Check if this is a killer move (beta cutoff move from previous iterations)
            if use_killer_moves:
                for depth_idx in range(min(depth, len(killer_moves))):
                    if killer_moves[depth_idx][0] == move or killer_moves[depth_idx][1] == move:
                        score += 1000  # High priority for killer moves

            if target:
                score += 10 * abs(piece_values.get(target, 0)) - abs(piece_values.get(piece, 0))

            score += (3 - abs(3.5 - move[1][1])) + (3 - abs(3.5 - move[1][0]))

            if piece[1] == 'P' and (move[1][0] == 0 or move[1][0] == 7):
                score += 900

            ordered_moves.append((move, score))

        return sorted(ordered_moves, key=lambda x: x[1], reverse=(color == 'W'))

    scored_moves = order_moves(board, legal_moves, color)

    length = len(scored_moves)
    if length > eval(depth_formula):
        scored_moves = scored_moves[:eval(depth_formula)]

    # DEBUG: Check if moves are being eliminated
    if display_simulation or len(scored_moves) == 0:
        print(f"DEBUG: depth={depth}, color={color}, legal_moves={len(legal_moves)}, scored_moves={len(scored_moves)}, formula={depth_formula}, eval={eval(depth_formula) if length > 0 else 'N/A'}")

    best_eval = float('-inf') if color == AI_color else float('inf')
    best_move_new = []

    for move_index, (move, _) in enumerate(scored_moves):
        new_board = simulate_move(board, move, real_board=False)

        # ======================================================================================
        # LATE MOVE REDUCTION OPTIMIZATION
        # ======================================================================================
        #
        # Assumes later moves in ordered list are less likely to be best moves.
        #
        # HOW IT WORKS:
        # 1. For moves after first 3-4 in ordered list, search at reduced depth (depth-2)
        # 2. If reduced-depth search doesn't improve alpha, skip full-depth search
        # 3. Based on principle: best moves are usually found early in ordering
        # 4. Only applied when not in check and not capturing (tactical moves need full depth)
        #
        # SIGNIFICANT SPEEDUP: Reduces branching factor for unpromising moves
        #
        # ======================================================================================
        # Late Move Reduction
        if depth >= 3 and move_index > 3 and not check and not board[move[1][0]][move[1][1]]:
            eval_board, opponent_best_move = select_best_ai_move_improved(new_board, depth-2, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            if (color == AI_color and eval_board < alpha) or (color != AI_color and eval_board > beta):
                continue

        eval_board, opponent_best_move = select_best_ai_move_improved(new_board, depth-1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)

        # Update bounds immediately with current move evaluation
        if color == AI_color:
            alpha = max(alpha, eval_board)
        else:
            beta = min(beta, eval_board)

        # Update best move if this evaluation is better
        if (color == AI_color and eval_board > best_eval) or (color != AI_color and eval_board < best_eval):
            best_eval = eval_board
            best_move_new = [move] + opponent_best_move if opponent_best_move else [move]

        if beta <= alpha:
            # Store killer move for better move ordering in future searches
            if use_killer_moves and depth < len(killer_moves):
                if killer_moves[depth][0] != move:
                    killer_moves[depth][1] = killer_moves[depth][0]  # Shift previous killer move
                    killer_moves[depth][0] = move  # Store new killer move
            break

    if use_transposition:
        transposition_table[board_key] = best_eval, depth, best_move_new
    
    # DEBUG: Print what we're returning if it's empty at top level
    if depth == initial_depth and len(best_move_new) == 0:
        print(f"WARNING at return: empty move! depth={depth}, color={color}, AI_color={AI_color}, legal={len(legal_moves)}, scored was={length}")
    
    return best_eval, best_move_new

# ======================================================================================
# QUIESCENCE SEARCH ALGORITHM
# ======================================================================================
#
# Solves the "horizon effect" where tactical threats are missed just beyond search depth.
#
# HOW IT WORKS:
# 1. At leaf nodes of main search, continues searching captures only
# 2. Ensures tactical stability - won't evaluate position where captures are still possible
# 3. Stops when no captures available or max depth reached
# 4. Uses "stand pat" evaluation - current position value if no capture is forced
# 5. Alpha-beta pruning applied to captures for efficiency
#
# MVV-LVA ORDERING:
# Sorts captures by Most Valuable Victim minus Least Valuable Attacker
# Prioritizes queen captures, then rook captures, etc.
# Example: Pawn takes Queen = high priority, Queen takes Pawn = lower priority
#
# ======================================================================================
def quiescence_search(board, color, AI_color, alpha, beta, depth, max_depth):
    stand_pat = evaluate_board(board)
    if depth >= max_depth:
        return stand_pat, []

    if color == 'W':
        if stand_pat >= beta:
            return beta, []
        if stand_pat > alpha:
            alpha = stand_pat
    else:
        if stand_pat <= alpha:
            return alpha, []
        if stand_pat < beta:
            beta = stand_pat

    captures = [move for move in get_all_legal_moves(board, color, check_legality=True) if board[move[1][0]][move[1][1]] != '']
    best_move = []

    # Sort captures by most valuable victim and least valuable attacker (MVV-LVA)
    def capture_value(move):
        attacker = board[move[0][0]][move[0][1]]
        victim = board[move[1][0]][move[1][1]]
        return piece_values.get(victim, 0) * 10 - piece_values.get(attacker, 0)

    captures.sort(key=capture_value, reverse=True)

    for move in captures:
        new_board = simulate_move(board, move, real_board=False)
        score, _ = quiescence_search(new_board, 'B' if color == 'W' else 'W', AI_color, -beta, -alpha, depth + 1, max_depth)
        score = -score
        if color == 'W':
            if score > alpha:
                alpha = score
                best_move = [move]
            if alpha >= beta:
                return beta, best_move
        else:
            if score < beta:
                beta = score
                best_move = [move]
            if beta <= alpha:
                return alpha, best_move

    return (alpha if color == 'W' else beta, best_move)








def convert_to_standard_notation_simple(move):
    """Convert a move from the game's format to simple standard chess notation (just coordinates)."""
    start, end = move
    start_square = chr(97 + start[1]) + str(8 - start[0])
    end_square = chr(97 + end[1]) + str(8 - end[0])
    return f"{start_square}{end_square}"

def parse_llm_response(response, board, color):
    """Parse the LLM response and convert all moves to the game's format."""
    moves = response.strip().split()
    if not moves:
        print("No moves found in response")
        return []

    parsed_moves = []
    for first_move in moves:
        first_move = first_move.lower()  # Convert to lowercase for consistent processing
        #print(f"Parsing move: {first_move}")

        # Handle castling
        if first_move in ['o-o', 'o-o-o', '0-0', '0-0-0']:
            row = 7 if color == 'W' else 0
            if first_move in ['o-o', '0-0']:  # Kingside castling
                move = ((row, 4), (row, 6))
            else:  # Queenside castling
                move = ((row, 4), (row, 2))
            print(f"Castling move: {move}")
            parsed_moves.append(move)
            continue

        # Handle pawn promotion
        promotion = None
        if '=' in first_move:
            first_move, promotion = first_move.split('=')
            promotion = promotion.upper()

        # Remove '+' or '#' if present (check or checkmate symbols)
        first_move = first_move.rstrip('+#')

        # Handle UCI notation (e.g., "e2e4", "e7e8q", "E2E4", "E7E8Q")
        if len(first_move) in [4, 5] and first_move[:4].isalnum():
            start_col, start_row = ord(first_move[0].lower()) - 97, 8 - int(first_move[1])
            end_col, end_row = ord(first_move[2].lower()) - 97, 8 - int(first_move[3])

            # Handle promotion (5th character in UCI notation)
            move_promotion = None
            if len(first_move) == 5:
                promo_char = first_move[4].lower()
                # Map UCI promotion chars to piece letters
                promo_map = {'q': 'Q', 'r': 'R', 'b': 'B', 'n': 'N'}
                if promo_char in promo_map:
                    move_promotion = promo_map[promo_char]
                else:
                    print(f"Warning: Invalid promotion character '{promo_char}' in move '{first_move}', ignoring promotion")

            move = ((start_row, start_col), (end_row, end_col))
            #print(f"UCI move: {move}, promotion: {move_promotion}")

            # Use move_promotion if present, otherwise fall back to the '=' parsed promotion
            final_promotion = move_promotion or promotion
            parsed_moves.append(move if not final_promotion else move + (final_promotion,))
            continue

        # Handle standard algebraic notation (e.g., "e4", "Nf3", "E4", "NF3")
        if len(first_move) in [2, 3, 4]:
            piece = 'P' if len(first_move) == 2 or first_move[0].islower() else first_move[0].upper()
            end_col = ord(first_move[-2].lower()) - 97
            if first_move[-1].isdigit():
                end_row = 8 - int(first_move[-1])
            else:
                print(f"Invalid move format: {first_move}")
                continue
            
            # Find the piece that can make this move
            for start_row in range(8):
                for start_col in range(8):
                    if board[start_row][start_col].upper() == color + piece:
                        move = ((start_row, start_col), (end_row, end_col))
                        print(f"Standard algebraic move: {move}")
                        parsed_moves.append(move if not promotion else move + (promotion,))
                        break
                else:
                    continue
                break
            else:
                print(f"Failed to parse move: {first_move}")

    #print(f"All parsed moves: {parsed_moves}")
    return parsed_moves


def initialize_ai_model(color):
    global llm_white, llm_black, block_size, ai_method_white, ai_method_black

    print(f"AI model initialized successfully for {color}.")
    if color == 'W':
        print("Initializing white model...")
        llm_white = brain_inference.initialize_model()
        if llm_white is None:
            print(f"Failed to initialize AI model for {color}. Exiting.")
            return False
        block_size = brain_inference.global_model.block_size # only allow one block size
        ai_method_white = "LLM"  # Automatically switch to LLM when model loads
    else:
        print("Initializing black model...")
        llm_black = brain_inference.initialize_model()
        if llm_black is None:
            print(f"Failed to initialize AI model for {color}. Exiting.")
            return False
        block_size = brain_inference.global_model.block_size
        ai_method_black = "LLM"  # Automatically switch to LLM when model loads

    return True
def select_best_ai_move_improved(board, depth, color, AI_color, alpha=float('-inf'), beta=float('inf'), display_simulation=False, last_move=None, initial_depth=None):
    global transposition_table, piece_values, depth_formula, discount

    # Use optimizations
    use_transposition = True
    use_killer_moves = True

    # Transposition table lookup
    board_key = (tuple(map(tuple, board)), color, AI_color)  # Include whose turn it is and which AI is playing
    if use_transposition and board_key in transposition_table:
        eval_board, eval_depth, best_move_new = transposition_table[board_key]
        if display_simulation:
           print(f"Transposition Table hit for {color} {depth} {eval_board:.2f} {eval_depth} {best_move_new} {last_move}")
        if eval_depth >= depth:
            return eval_board, best_move_new

    legal_moves = get_all_legal_moves(board, color, last_move=last_move, check_legality=True)
    
    # DEBUG: Check if legal_moves is unexpectedly empty at top level
    if depth == initial_depth:
        print(f"DEBUG ENTRY: color={color}, depth={depth}, initial={initial_depth}, legal_moves={len(legal_moves)}")
    
    check = is_in_check(board, color)

    evaluation = evaluate_board(board)

    if not check and not legal_moves:
        evaluation = evaluation * (discount**((initial_depth + 1) - depth))
        if display_simulation:
            print(f"Stalemate, no legal moves for {color}. Evaluation: {-evaluation:.2f}")
            print(f"Depth: {depth}, Last move: {last_move}")
        return -evaluation, []

    if check and not legal_moves:
        if color == "W":
            evaluation = (evaluation - 100) * (discount**((initial_depth + 1) - depth))
        else:
            evaluation = (evaluation + 100) * (discount**((initial_depth + 1) - depth))
        if display_simulation:
            print(f"Checkmate of {color}. Evaluation: {evaluation:.2f}")
            print(f"Depth: {depth}, Last move: {last_move} of {'W' if color == 'B' else 'B'}")
        return evaluation, []

    if depth <= 0:
        return quiescence_search(board, color, AI_color, alpha, beta, 0, 2)

    # ======================================================================================
    # NULL MOVE PRUNING OPTIMIZATION
    # ======================================================================================
    #
    # Allows opponent to "pass" their turn to test position strength.
    #
    # HOW IT WORKS:
    # 1. Temporarily give opponent an extra move (null move)
    # 2. If opponent still can't create a threat, position must be very strong
    # 3. Reduces search tree by pruning weak positions early
    # 4. Only used when not in check (passing when in check is illegal)
    # 5. Uses reduced depth (depth-3) for efficiency
    #
    # EXAMPLE: If current position is so good that opponent gains nothing from extra move,
    # then we can prune this branch and avoid deeper search.
    #
    # ======================================================================================
    # Null Move Pruning (not at root level - must return actual move at root)
    if depth > 2 and not check and depth < initial_depth:
        null_move_eval, _ = select_best_ai_move_improved(board, depth - 3, 'B' if color == 'W' else 'W', AI_color, -beta, -beta + 1, display_simulation, last_move, initial_depth)
        null_move_eval = -null_move_eval
        if null_move_eval >= beta:
            return beta, []

    # ======================================================================================
    # MOVE ORDERING HEURISTIC - CRITICAL FOR ALPHA-BETA EFFICIENCY
    # ======================================================================================
    #
    # Orders moves to improve alpha-beta pruning effectiveness.
    # Better ordered moves = more pruning = faster search.
    #
    # ORDERING CRITERIA (highest priority first):
    # 1. Killer Moves: Moves that caused beta cutoffs at same depth in other branches
    # 2. Captures: MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    #    - Queen takes Pawn (high priority - saves queen)
    #    - Pawn takes Queen (high priority - gains queen)
    # 3. Center control and advancement bonuses
    # 4. Pawn promotions (very high priority)
    #
    # Killer moves work because: if a move caused cutoff elsewhere, likely good here too
    #
    # ======================================================================================
    # Move Ordering with killer moves heuristic
    def order_moves(board, moves, color):
        global killer_moves
        ordered_moves = []
        for move in moves:
            score = 0
            piece = board[move[0][0]][move[0][1]]
            target = board[move[1][0]][move[1][1]]

            # Check if this is a killer move (beta cutoff move from previous iterations)
            if use_killer_moves:
                for depth_idx in range(min(depth, len(killer_moves))):
                    if killer_moves[depth_idx][0] == move or killer_moves[depth_idx][1] == move:
                        score += 1000  # High priority for killer moves

            if target:
                score += 10 * abs(piece_values.get(target, 0)) - abs(piece_values.get(piece, 0))

            score += (3 - abs(3.5 - move[1][1])) + (3 - abs(3.5 - move[1][0]))

            if piece[1] == 'P' and (move[1][0] == 0 or move[1][0] == 7):
                score += 900

            ordered_moves.append((move, score))

        return sorted(ordered_moves, key=lambda x: x[1], reverse=(color == 'W'))

    scored_moves = order_moves(board, legal_moves, color)

    length = len(scored_moves)
    if length > eval(depth_formula):
        scored_moves = scored_moves[:eval(depth_formula)]

    # DEBUG: Check if moves are being eliminated
    if display_simulation or len(scored_moves) == 0:
        print(f"DEBUG: depth={depth}, color={color}, legal_moves={len(legal_moves)}, scored_moves={len(scored_moves)}, formula={depth_formula}, eval={eval(depth_formula) if length > 0 else 'N/A'}")

    best_eval = float('-inf') if color == AI_color else float('inf')
    best_move_new = []

    for move_index, (move, _) in enumerate(scored_moves):
        new_board = simulate_move(board, move, real_board=False)

        # ======================================================================================
        # LATE MOVE REDUCTION OPTIMIZATION
        # ======================================================================================
        #
        # Assumes later moves in ordered list are less likely to be best moves.
        #
        # HOW IT WORKS:
        # 1. For moves after first 3-4 in ordered list, search at reduced depth (depth-2)
        # 2. If reduced-depth search doesn't improve alpha, skip full-depth search
        # 3. Based on principle: best moves are usually found early in ordering
        # 4. Only applied when not in check and not capturing (tactical moves need full depth)
        #
        # SIGNIFICANT SPEEDUP: Reduces branching factor for unpromising moves
        #
        # ======================================================================================
        # Late Move Reduction
        if depth >= 3 and move_index > 3 and not check and not board[move[1][0]][move[1][1]]:
            eval_board, opponent_best_move = select_best_ai_move_improved(new_board, depth-2, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            if (color == AI_color and eval_board < alpha) or (color != AI_color and eval_board > beta):
                continue

        eval_board, opponent_best_move = select_best_ai_move_improved(new_board, depth-1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)

        # Update bounds immediately with current move evaluation
        if color == AI_color:
            alpha = max(alpha, eval_board)
        else:
            beta = min(beta, eval_board)

        # Update best move if this evaluation is better
        if (color == AI_color and eval_board > best_eval) or (color != AI_color and eval_board < best_eval):
            best_eval = eval_board
            best_move_new = [move] + opponent_best_move if opponent_best_move else [move]

        if beta <= alpha:
            # Store killer move for better move ordering in future searches
            if use_killer_moves and depth < len(killer_moves):
                if killer_moves[depth][0] != move:
                    killer_moves[depth][1] = killer_moves[depth][0]  # Shift previous killer move
                    killer_moves[depth][0] = move  # Store new killer move
            break

    if use_transposition:
        transposition_table[board_key] = best_eval, depth, best_move_new
    
    # DEBUG: Print what we're returning if it's empty at top level
    if depth == initial_depth and len(best_move_new) == 0:
        print(f"WARNING at return: empty move! depth={depth}, color={color}, AI_color={AI_color}, legal={len(legal_moves)}, scored was={length}")
    
    return best_eval, best_move_new

# ======================================================================================
# QUIESCENCE SEARCH ALGORITHM
# ======================================================================================
#
# Solves the "horizon effect" where tactical threats are missed just beyond search depth.
#
# HOW IT WORKS:
# 1. At leaf nodes of main search, continues searching captures only
# 2. Ensures tactical stability - won't evaluate position where captures are still possible
# 3. Stops when no captures available or max depth reached
# 4. Uses "stand pat" evaluation - current position value if no capture is forced
# 5. Alpha-beta pruning applied to captures for efficiency
#
# MVV-LVA ORDERING:
# Sorts captures by Most Valuable Victim minus Least Valuable Attacker
# Prioritizes queen captures, then rook captures, etc.
# Example: Pawn takes Queen = high priority, Queen takes Pawn = lower priority
#
# ======================================================================================
def quiescence_search(board, color, AI_color, alpha, beta, depth, max_depth):
    stand_pat = evaluate_board(board)
    if depth >= max_depth:
        return stand_pat, []

    if color == 'W':
        if stand_pat >= beta:
            return beta, []
        if stand_pat > alpha:
            alpha = stand_pat
    else:
        if stand_pat <= alpha:
            return alpha, []
        if stand_pat < beta:
            beta = stand_pat

    captures = [move for move in get_all_legal_moves(board, color, check_legality=True) if board[move[1][0]][move[1][1]] != '']
    best_move = []

    # Sort captures by most valuable victim and least valuable attacker (MVV-LVA)
    def capture_value(move):
        attacker = board[move[0][0]][move[0][1]]
        victim = board[move[1][0]][move[1][1]]
        return piece_values.get(victim, 0) * 10 - piece_values.get(attacker, 0)

    captures.sort(key=capture_value, reverse=True)

    for move in captures:
        new_board = simulate_move(board, move, real_board=False)
        score, _ = quiescence_search(new_board, 'B' if color == 'W' else 'W', AI_color, -beta, -alpha, depth + 1, max_depth)
        score = -score
        if color == 'W':
            if score > alpha:
                alpha = score
                best_move = [move]
            if alpha >= beta:
                return beta, best_move
        else:
            if score < beta:
                beta = score
                best_move = [move]
            if beta <= alpha:
                return alpha, best_move

    return (alpha if color == 'W' else beta, best_move)








def convert_to_standard_notation_simple(move):
    """Convert a move from the game's format to simple standard chess notation (just coordinates)."""
    start, end = move
    start_square = chr(97 + start[1]) + str(8 - start[0])
    end_square = chr(97 + end[1]) + str(8 - end[0])
    return f"{start_square}{end_square}"

def parse_llm_response(response, board, color):
    """Parse the LLM response and convert all moves to the game's format."""
    moves = response.strip().split()
    if not moves:
        print("No moves found in response")
        return []

    parsed_moves = []
    for first_move in moves:
        first_move = first_move.lower()  # Convert to lowercase for consistent processing
        #print(f"Parsing move: {first_move}")

        # Handle castling
        if first_move in ['o-o', 'o-o-o', '0-0', '0-0-0']:
            row = 7 if color == 'W' else 0
            if first_move in ['o-o', '0-0']:  # Kingside castling
                move = ((row, 4), (row, 6))
            else:  # Queenside castling
                move = ((row, 4), (row, 2))
            print(f"Castling move: {move}")
            parsed_moves.append(move)
            continue

        # Handle pawn promotion
        promotion = None
        if '=' in first_move:
            first_move, promotion = first_move.split('=')
            promotion = promotion.upper()

        # Remove '+' or '#' if present (check or checkmate symbols)
        first_move = first_move.rstrip('+#')

        # Handle UCI notation (e.g., "e2e4", "e7e8q", "E2E4", "E7E8Q")
        if len(first_move) in [4, 5] and first_move[:4].isalnum():
            start_col, start_row = ord(first_move[0].lower()) - 97, 8 - int(first_move[1])
            end_col, end_row = ord(first_move[2].lower()) - 97, 8 - int(first_move[3])

            # Handle promotion (5th character in UCI notation)
            move_promotion = None
            if len(first_move) == 5:
                promo_char = first_move[4].lower()
                # Map UCI promotion chars to piece letters
                promo_map = {'q': 'Q', 'r': 'R', 'b': 'B', 'n': 'N'}
                if promo_char in promo_map:
                    move_promotion = promo_map[promo_char]
                else:
                    print(f"Warning: Invalid promotion character '{promo_char}' in move '{first_move}', ignoring promotion")

            move = ((start_row, start_col), (end_row, end_col))
            #print(f"UCI move: {move}, promotion: {move_promotion}")

            # Use move_promotion if present, otherwise fall back to the '=' parsed promotion
            final_promotion = move_promotion or promotion
            parsed_moves.append(move if not final_promotion else move + (final_promotion,))
            continue

        # Handle standard algebraic notation (e.g., "e4", "Nf3", "E4", "NF3")
        if len(first_move) in [2, 3, 4]:
            piece = 'P' if len(first_move) == 2 or first_move[0].islower() else first_move[0].upper()
            end_col = ord(first_move[-2].lower()) - 97
            if first_move[-1].isdigit():
                end_row = 8 - int(first_move[-1])
            else:
                print(f"Invalid move format: {first_move}")
                continue
            
            # Find the piece that can make this move
            for start_row in range(8):
                for start_col in range(8):
                    if board[start_row][start_col].upper() == color + piece:
                        move = ((start_row, start_col), (end_row, end_col))
                        print(f"Standard algebraic move: {move}")
                        parsed_moves.append(move if not promotion else move + (promotion,))
                        break
                else:
                    continue
                break
            else:
                print(f"Failed to parse move: {first_move}")

    #print(f"All parsed moves: {parsed_moves}")
    return parsed_moves


def initialize_ai_model(color):
    global llm_white, llm_black, block_size, ai_method_white, ai_method_black

    print(f"AI model initialized successfully for {color}.")
    if color == 'W':
        print("Initializing white model...")
        llm_white = brain_inference.initialize_model()
        if llm_white is None:
            print(f"Failed to initialize AI model for {color}. Exiting.")
            return False
        block_size = brain_inference.global_model.block_size # only allow one block size
        ai_method_white = "LLM"  # Automatically switch to LLM when model loads
    else:
        print("Initializing black model...")
        llm_black = brain_inference.initialize_model()
        if llm_black is None:
            print(f"Failed to initialize AI model for {color}. Exiting.")
            return False
        block_size = brain_inference.global_model.block_size
        ai_method_black = "LLM"  # Automatically switch to LLM when model loads

    return True
def select_best_ai_move_llm(board, depth, color, AI_color, alpha=float('-inf'), beta=float('inf'), display_simulation=False, last_move=None, initial_depth=None):
    """AI method that uses the LLM for move selection."""
    global game_history_simple, llm_white, llm_black
    k = 10
    
    if color == 'W':
        model = llm_white
    else:
        model = llm_black
    
    if model is None:
        print(f"No AI model loaded for {color}. Please initialize the AI model first.")
        return None, []

    # Prepare the input for the LLM
    if len(game_history_simple) == 1:
        moves_string = game_history_simple[0]  # This will be "<startgame>"
    elif len(game_history_simple) == 2:
        moves_string = "".join(game_history_simple)  # "<startgame> FirstMove"
    else:
        moves_string = " ".join(game_history_simple)

    # Truncate the moves_string to fit within the model's block size
    # CHANGES: Remove whole moves and special tokens in chunks
    while len(moves_string) > block_size:
        if moves_string.startswith("<STARTGAME>"):
            moves_string = moves_string[11:]  # Remove "<STARTGAME>"
        else:
            space_index = moves_string.find(' ')
            if space_index != -1:
                moves_string = moves_string[space_index + 1:]
            else:
                moves_string = moves_string[-block_size:]
    
    prompt = f"Chess game in progress. Move history: {moves_string} Move: {color}"
    print("Prompt: ", prompt)

    # Get the LLM's response
    token_to_idx = brain_inference.global_tokenizer
    #char_to_idx = brain_inference.global_char_to_idx
    idx_to_token = brain_inference.global_tokenizer_reverse

    use_characters = brain_inference.global_use_characters
    use_chess_moves = brain_inference.global_use_chess_moves
    response_length = 10

    #def generate_response(model, tokenizer, tokenizer_reverse, input_text, tokens_to_generate=10, use_characters=False, use_chess_moves=True, top_k=k):
    try:
        llm_responses = generate_response(model, token_to_idx, idx_to_token, moves_string, tokens_to_generate=response_length, use_characters=use_characters, use_chess_moves=use_chess_moves, top_k=k)
        print("LLM responses: ", llm_responses)
    except IndexError:
        print("Error: LLM failed to generate responses. Falling back to random move.")
        llm_responses = []

    legal_moves = get_all_legal_moves(board, color, last_move=last_move, check_legality=True)

    suggested_moves = []  # Initialize to avoid UnboundLocalError

    for response_k, response in enumerate(llm_responses)    :
        #print("Response: ", response)
       
        response_move = response
        #print("Response move: ", response_move)
        suggested_moves = parse_llm_response(response_move, board, color)
        #print("Suggested moves: ", suggested_moves)

        if suggested_moves:
            first_move = suggested_moves[0]
            if first_move in legal_moves:
                print(f"First move in response number {response_k+1} is a legal move: {first_move}")
                # Track LLM statistics
                llm_stats['total_moves'] += 1
                if response_k + 1 == 1:
                    llm_stats['first_legal'] += 1
                elif response_k + 1 == 2:
                    llm_stats['second_legal'] += 1
                else:  # 3rd or higher
                    llm_stats['third_plus_legal'] += 1
                new_board = simulate_move(board, first_move)
                evaluation = evaluate_board(new_board)
                return evaluation, suggested_moves, response_k+1  # Return all suggested moves for visualization
    
    # If no legal move is found in the top K responses, fall back to the best evaluated legal move
    llm_stats['total_moves'] += 1
    llm_stats['no_legal'] += 1

    if legal_moves:
        if AI_color == 'W':
            fallback_move = max(legal_moves, key=lambda move: evaluate_board(simulate_move(board, move)))
        else:
            fallback_move = min(legal_moves, key=lambda move: evaluate_board(simulate_move(board, move)))
    else:
        fallback_move = None

    if fallback_move:
        print("Fallback move from legal moves: ", fallback_move)
        new_board = simulate_move(board, fallback_move)
        evaluation = evaluate_board(new_board)
        # Include fallback move and any additional suggested moves if available
        additional_moves = suggested_moves[1:] if len(suggested_moves) > 1 else []
        return evaluation, [fallback_move] + additional_moves
    else:
        return evaluate_board(board), []

# Update the convert_to_standard_notation function to remove 'P' for pawns
def convert_to_standard_notation(board, move):
    """Convert a move from the game's format to standard chess notation."""
    start, end = move
    piece = board[start[0]][start[1]]
    piece_type = piece[1] if piece and len(piece) > 1 and piece[1] != 'P' else ''
    start_square = chr(97 + start[1]) + str(8 - start[0])
    end_square = chr(97 + end[1]) + str(8 - end[0])
    return f"{piece_type}{start_square}{end_square}"

# ======================================================================================
# POSITION EVALUATION - MATERIAL + PIECE POSITIONS
# ======================================================================================
#
# The best balance of speed and accuracy for chess evaluation.
# Considers both material value and optimal piece positioning.
#
# ======================================================================================

# ======================================================================================
# POSITION EVALUATION FUNCTION
# ======================================================================================
#
# Assigns numerical value to chess position from White's perspective.
#
# MATERIAL VALUES:
# Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9, King=100
# (King value prevents trading king for pieces)
#
# POSITIONAL BONUSES (piece-square tables):
# - Knights: Better in center (d4,e4,d5,e5 squares)
# - Bishops: Prefer open diagonals, center control
# - Rooks: 7th rank bonus, open files
# - Queens: Center preference, mobility
# - Kings: Safety in corners during middlegame, center in endgame
# - Pawns: Advance toward promotion, avoid isolation
#
# CALCULATION:
# Total = Material difference + Positional bonuses
# Positive = White advantage, Negative = Black advantage
#
# ======================================================================================
# Evaluate the board for given state (basic evaluation without consideration of checkmate or check)
def evaluate_board_positions_optimized(board):
    global piece_values, get_pos_val_white, get_pos_val_black

    evaluation = 0

    # Use globally precomputed positional values for maximum speed
    for i in range(8):
        row = board[i]
        for j in range(8):
            piece = row[j]
            if piece:
                is_white = piece[0] == 'W'
                piece_type = piece[1]
                value = piece_values.get(piece, 0)
                pos_val = get_pos_val_white.get((piece_type, i, j), 0) if is_white else get_pos_val_black.get((piece_type, i, j), 0)
                evaluation += value + pos_val

    return evaluation



def draw_board(screen):
    font = pygame.font.SysFont("Arial", 12)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            # The color pattern depends on the original row and col
            color = BLUE if (row + col) % 2 == 0 else GRAY

            # Calculate the display positions based on board_reversed
            display_row = BOARD_SIZE - 1 - row if board_reversed else row
            display_col = BOARD_SIZE - 1 - col if board_reversed else col

            square = pygame.Rect(display_col * SQUARE_SIZE, display_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, square)

            # Adjust the coordinate display
            square_label_row = 8 - row #if not board_reversed else row + 1
            square_label_col = string.ascii_lowercase[col] #if not board_reversed else string.ascii_lowercase[BOARD_SIZE - 1 - col]
            coords_text = font.render(f"{square_label_col}{square_label_row}", True, BLACK)
            screen.blit(coords_text, (square.right - coords_text.get_width(),
                                      square.bottom - coords_text.get_height()))
            
def draw_pieces(screen, board):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece:
                # Calculate the display positions based on board_reversed
                display_row = BOARD_SIZE - 1 - row if board_reversed else row
                display_col = BOARD_SIZE - 1 - col if board_reversed else col

                if piece[0] == "W":
                    text = font_2big.render(pieces_uni[piece], True, WHITE)
                else:
                    text = font_2big.render(pieces_uni[piece], True, BLACK)
                screen.blit(text, (display_col * SQUARE_SIZE + 8, display_row * SQUARE_SIZE - 8))

# Replay AI Thought Process
def play_moves(board, moves, max_depth):
    global show_simulation
    show_simulation = True
    print(f"Max Depth: {max_depth}, Playing moves:", moves)
    draw_board_wrapper(screen, board)
    pygame.display.flip()
    
    if not moves:
        print("No moves to play")
        return board
    
    time_delay = 2000  # Set a fixed delay of 2 seconds
    
    # Handle the case where moves is a single move (iterative deepening)
    if isinstance(moves, tuple) and len(moves) == 2:
        moves = [moves]
    
    for i, move in enumerate(moves):
        print(f"Processing move {i+1}/{len(moves)}: {move}")  # Debug print
        
        if isinstance(move, str) and "End" in move:
            print(f"Search ended: {move}")
            break
        
        if isinstance(move, tuple) and len(move) == 2:
            start_pos, end_pos = move
            piece = board[start_pos[0]][start_pos[1]]
            if piece:
                print(f"Move {i+1}: Piece {piece} from {start_pos} to {end_pos}")
                
                notation = convert_to_chess_notation(board, move)
                print(f"Move: {notation}")
                read_aloud(f"{notation}")
                
                board = simulate_move(board, move, real_board=True)
                draw_board_wrapper(screen, board)
                
                pygame.display.flip()
                pygame.time.wait(time_delay)
            else:
                print(f"Skipping move {i+1}: No piece at {start_pos}")
        else:
            print(f"Unexpected move format: {move}")
    
    print("Finished playing all moves")  # Debug print
    show_simulation = False    
    pygame.time.wait(time_delay)
    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))
    screen.blit(font_info.render(f"Move: {move_number}. Player: {player}.  Ai: {ai_method}. {depth_equation}", True, BLACK), (27, 650))
    return board


def convert_to_chess_notation(board, move):
    move_notation = ""
    if len(move) == 2:
        start, end = move
        start_row, start_col = start
        end_row, end_col = end
        piece = board[start_row][start_col]
        if piece !="":
            piece_name = piece_dict[piece]
            start_square = string.ascii_lowercase[start_col] + str(8 - start_row)
            end_square = string.ascii_lowercase[end_col] + str(8 - end_row)
            move_notation = piece_name + " from " + start_square + " to " + end_square
    return move_notation


def read_aloud(text):
    if sound_enabled:
        if platform.system() == "Darwin":  # Check if the system is macOS
            os.system("say " + text)
        else:
            pass
    else:
        print("Text-to-speech (disabled): " + text)
        

def initialize_game():
    # Initial game state setup
    global setauto_switch_colors_for_player, depth_formula,transposition_table, depth_equation,discount,player_turn, selected_piece, actual_last_move, \
        list_of_boards, move_number, end_of_game, running, show_simulation, board, depth, player,ai, evaluate_board, evaluation_method, \
            select_best_ai_move, ai_method, has_moved, auto_save, game_history, game_history_simple, position_history, board_reversed, sound_enabled, has_moved_history
    # The initial board setup, simplified without pawn promotion
    board = initial_board
    player_turn = True
    selected_piece = None
    actual_last_move = None  # Will hold the last move made in the game as a tuple: ((start_row, start_col), (end_row, end_col))
    list_of_boards = [copy.deepcopy(board)]  # Start with just initial position
    move_number = 0
    player_turn = True
    selected_piece = None
    end_of_game = False
    running = True
    show_simulation = False  # Default: don't show AI thinking for faster gameplay
    end_of_game = False
    depth = 3
    transposition_table = {}
    player = "W"
    ai = "B"
    player_turn = True
    setauto_switch_colors_for_player = False
    auto_save = False

    # PRESERVE AI method settings - DO NOT reset if LLM models are loaded
    # Only reset to default if no LLM is loaded
    if llm_white is None:
        ai_method_white = "Improved"
    if llm_black is None:
        ai_method_black = "Improved"
    # Preserve block_size if models are loaded
    if llm_white is None and llm_black is None:
        block_size = None
   
    sound_enabled = True
    board_reversed = False
    evaluation_method = "Positions_optimized"
    evaluate_board = evaluation_methods[evaluation_method]
    ai_method = "Improved"
    select_best_ai_move = ai_methods[ai_method]
    depth_equation = "All .95"
    depth_formula = depth_equations[depth_equation][0]
    discount = depth_equations[depth_equation][1]

    has_moved = {"WK": False, "WR1": False, "WR2": False, "BK": False, "BR1": False, "BR2": False}
    has_moved_history = [copy.deepcopy(has_moved)]

    game_history = ["<STARTGAME> "]          # Start with just start token
    game_history_simple = ["<STARTGAME>"]    # Start with just start token
    position_history = []                     # Empty initially
    screen.fill(WHITE)
    draw_board_wrapper(screen, board)
    draw_pieces_not_on_board(screen,board, height=SCREEN_HEIGHT)
    pygame.display.set_caption("JMR's Game of Chess")
    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,200))
    screen.blit(font_info.render(f"Move: {move_number}. Player: {player}.  Ai: {ai_method}. {depth_equation}", True, BLACK), (27, SCREEN_HEIGHT - 150))
    screen.blit(font_info.render(f"Depth: {depth}. Evaluation Method {evaluation_method}. Show simulation: {show_simulation}", True, BLACK), (27, SCREEN_HEIGHT - 125))
    pygame.display.flip()

    # LLM models are loaded on-demand when switching to LLM mode

    # Reset LLM move statistics
    llm_stats = {
        'total_moves': 0,
        'first_legal': 0,
        'second_legal': 0,
        'third_plus_legal': 0,
        'no_legal': 0
    }


ai_methods = {
    'LLM': select_best_ai_move_llm,
    'Improved': select_best_ai_move_improved,
}

evaluation_methods = {
    'Positions_optimized': evaluate_board_positions_optimized,
}

depth_equations = {
    "All no discount": ("length", 1),
    "All .95": ("length", .95),
    "All .90": ("length", .90),
    "Half length in 2 .95": ("int(length/2) if depth < initial_depth - 2 and length > 4 else length", .95),
    "Half length in 4 .95": ("int(length/2) if depth < initial_depth - 4 and length > 4 else length", .95),
    "length/initial_depth +1-depth in 4 .95": ("int(length/((initial_depth +1)-depth)) if depth < initial_depth - 4 and length > 4 else length", .90),
    }

def help():
    # Display condensed help in bottom status area (like move messages)
    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50, 200))

    # Use smaller font for help
    small_font = pygame.font.SysFont("Arial", 18)
    y = SCREEN_HEIGHT - 200  # Move up 30 pixels

    screen.blit(small_font.render("Mouse: Click piece then destination to move", True, BLACK), (27, y))
    y += 16
    screen.blit(small_font.render("s: save, l: load, r: restart", True, BLACK), (27, y))
    y += 16
    screen.blit(small_font.render("x: switch player side", True, BLACK), (27, y))
    y += 16
    screen.blit(small_font.render("y: toggle AI thinking display", True, BLACK), (27, y))
    y += 16
    screen.blit(small_font.render("Up/Down: change AI depth (3=default)", True, BLACK), (27, y))
    y += 16
    screen.blit(small_font.render("a: cycle WHITE AI, z: cycle BLACK AI", True, BLACK), (27, y))
    y += 16
    screen.blit(small_font.render("m: reload LLM for non-playing side", True, BLACK), (27, y))
    y += 16
    screen.blit(small_font.render("v: toggle self-play (AI vs AI)", True, BLACK), (27, y))
    y += 16
    screen.blit(small_font.render("Left/Right: review move history", True, BLACK), (27, y))
    y += 16
    screen.blit(small_font.render("d: cycle depth equations", True, BLACK), (27, y))
    y += 16
    screen.blit(small_font.render("Press any key to exit help", True, BLACK), (27, y))

    pygame.display.flip()

    # Wait for any key press to exit
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # Clear entire screen with white when exiting help
                screen.fill(WHITE)
                pygame.display.flip()
                waiting = False
                break
            elif event.type == pygame.QUIT:
                waiting = False
                break

# Add these dictionaries outside the help function:
player = "W"


# Killer moves heuristic for better move ordering
killer_moves = [[None, None] for _ in range(20)]  # Two killer moves per depth

# ======================================================================================
# ADVANCED CHESS AI ALGORITHMS - EXPLANATION
# ======================================================================================
#
# This chess engine implements several sophisticated algorithms for optimal move selection:
#
# 1. MINIMAX WITH ALPHA-BETA PRUNING:
#    The core search algorithm that explores game tree by maximizing/minimizing evaluation scores.
#    Alpha-beta pruning eliminates branches that won't affect the final decision, providing
#    30-60% speedup through the transposition table optimization.
#
# 2. TRANSPOSITION TABLE:
#    A persistent cache storing previously evaluated board positions with their scores and depths.
#    Prevents re-evaluating identical positions, providing major speed improvements.
#    Limited to 10,000 entries to prevent memory issues.
#
# 3. QUIESCENCE SEARCH:
#    Extends search in tactical positions (captures) to avoid "horizon effect" where
#    the engine misses tactical threats just beyond current search depth.
#    Uses MVV-LVA (Most Valuable Victim - Least Valuable Attacker) ordering for efficiency.
#
# 4. MOVE ORDERING HEURISTICS:
#    - Killer Moves: Remembers moves that caused beta cutoffs in previous searches
#    - MVV-LVA: Prioritizes captures of valuable pieces by less valuable attackers
#    - Center control and piece advancement bonuses
#    - Pawn promotion priority
#
# 5. NULL MOVE PRUNING:
#    Allows opponent to "pass" a turn to test if current position is so good that
#    even giving opponent an extra move still maintains advantage.
#
# 6. LATE MOVE REDUCTION:
#    Reduces search depth for moves considered later in the ordering, assuming
#    earlier moves are more likely to be best and need full-depth analysis.
#
# 7. POSITIONAL EVALUATION:
#    Uses piece-square tables that assign position-dependent values to pieces.
#    For example, knights are better in center, kings safer in corners during middlegame.
#
# 8. SPECIAL MOVE HANDLING:
#    - Castling: King moves 2 squares, rook jumps to other side, with attack checks
#    - En Passant: Pawn captures diagonally past adjacent pawn that just moved 2 squares
#    - Pawn Promotion: Pawns reaching 8th rank promote (knight if leads to checkmate)
#
# 9. DRAW DETECTION:
#    - Threefold repetition (player can claim)
#    - Fivefold repetition (automatic draw)
#    - Stalemate (no legal moves, not in check)
#
# ======================================================================================

# Main game loop
initialize_game()
selected_move = None
game_history_simple = ["<STARTGAME>"]   

# Global variables at start of game
saved_game_history = None
saved_game_history_simple = None
saved_list_of_boards = None
saved_position_history = None
saved_has_moved_history = None

# Add this variable to your game state
current_branch_point = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_UP:
                if depth < 21:
                    depth += 1
                    print("Depth set to", depth)
            elif event.key == pygame.K_DOWN:
                if depth > 0:
                    depth -= 1
                    print("Depth set to", depth)

            if event.key == pygame.K_y:
                show_simulation = not show_simulation
                if show_simulation:
                    print("AI Thinking will be displayed.")
                    read_aloud("AI Thinking will be displayed")
                else:
                    print("AI Thinking will not be displayed.")
                    read_aloud("AI Thinking will not be displayed")
            

            if event.key == pygame.K_LEFT:
                if move_number > 0:
                    # First time moving backwards, save the full histories
                    if saved_game_history is None:
                        saved_game_history = game_history.copy()
                        saved_game_history_simple = game_history_simple.copy()
                        saved_list_of_boards = list_of_boards.copy()
                        saved_position_history = position_history.copy()
                        saved_has_moved_history = has_moved_history.copy()
                        print("\nSaved full history:")
                        print(f"- Saved game history: {saved_game_history_simple}")
                    
                    move_number -= 1
                    print(f"\nMoving back to move_number: {move_number}")
                    
                    # Truncate current histories to the move we're viewing
                    game_history = game_history[:move_number * 2 + 1]
                    game_history_simple = game_history_simple[:move_number * 2 + 1]
                    list_of_boards = list_of_boards[:move_number + 1]
                    position_history = position_history[:move_number * 2 + 1]
                    has_moved_history = has_moved_history[:move_number + 1]
                    
                    board = copy.deepcopy(list_of_boards[move_number])
                    has_moved = copy.deepcopy(has_moved_history[move_number])
                    
                    player_turn = True
                    end_of_game = False

            elif event.key == pygame.K_RIGHT:
                # Only allow moving forward if we have saved history and haven't reached its end
                if saved_list_of_boards and move_number < len(saved_list_of_boards) - 1:
                    move_number += 1
                    print(f"\nMoving forward to move_number: {move_number}")
                    
                    # Restore from saved histories up to the current move_number
                    game_history = saved_game_history[:move_number * 2 + 1].copy()
                    game_history_simple = saved_game_history_simple[:move_number * 2 + 1].copy()
                    list_of_boards = saved_list_of_boards[:move_number + 1].copy()
                    position_history = saved_position_history[:move_number * 2 + 1].copy()
                    has_moved_history = saved_has_moved_history[:move_number + 1].copy()
                    
                    board = copy.deepcopy(saved_list_of_boards[move_number])
                    has_moved = copy.deepcopy(saved_has_moved_history[move_number])
                    
                    player_turn = True
                    if move_number == len(saved_list_of_boards) - 1:
                        # Clear saved histories when we reach the end of the saved game
                        saved_game_history = None
                        saved_game_history_simple = None
                        saved_list_of_boards = None
                        saved_position_history = None
                        saved_has_moved_history = None
                        if end_of_game:
                            print("Reached end of game")
                    else:
                        end_of_game = False


             #chaning AI method, moving through the dictionary, each time you press the key a key
            if event.key == pygame.K_a:
                # Cycle AI for current player
                keys = list(ai_methods.keys())
                if player == "W":
                    index = keys.index(ai_method_white)
                    if index < len(keys) - 1:
                        index += 1
                    else:
                        index = 0
                    ai_method_white = keys[index]
                    print(f"White AI method is now {ai_method_white}.")
                    # Initialize LLM if switched to LLM mode
                    if ai_method_white == "LLM" and llm_white is None:
                        print("Initializing LLM for White...")
                        initialize_ai_model('W')
                else:
                    index = keys.index(ai_method_black)
                    if index < len(keys) - 1:
                        index += 1
                    else:
                        index = 0
                    ai_method_black = keys[index]
                    print(f"Black AI method is now {ai_method_black}.")
                    # Initialize LLM if switched to LLM mode
                    if ai_method_black == "LLM" and llm_black is None:
                        print("Initializing LLM for Black...")
                        initialize_ai_model('B')

            if event.key == pygame.K_z:
                # Cycle AI for opponent
                keys = list(ai_methods.keys())
                if player == "W":
                    index = keys.index(ai_method_black)
                    if index < len(keys) - 1:
                        index += 1
                    else:
                        index = 0
                    ai_method_black = keys[index]
                    print(f"Black AI method is now {ai_method_black}.")
                    # Initialize LLM if switched to LLM mode
                    if ai_method_black == "LLM" and llm_black is None:
                        print("Initializing LLM for Black...")
                        initialize_ai_model('B')
                else:
                    index = keys.index(ai_method_white)
                    if index < len(keys) - 1:
                        index += 1
                    else:
                        index = 0
                    ai_method_white = keys[index]
                    print(f"White AI method is now {ai_method_white}.")
                    # Initialize LLM if switched to LLM mode
                    if ai_method_white == "LLM" and llm_white is None:
                        print("Initializing LLM for White...")
                        initialize_ai_model('W')

            if event.key == pygame.K_m:
                #initialize LLM for the non-playing color
                color = "B" if player == "W" else "W"
                initialize_ai_model(color)
                print(f"AI model reloaded for {color}.")

            if event.key == pygame.K_n:  # 'n' for noise
                sound_enabled = not sound_enabled
                status = "enabled" if sound_enabled else "disabled"
                read_aloud("Sound " + status)
                print(f"Sound {status}")

            if event.key == pygame.K_d:
                keys = list(depth_equations.keys())
                index = keys.index(depth_equation)
                if index < len(keys) - 1:
                    index += 1
                else:
                    index = 0
                depth_equation = keys[index]
                depth_formula = depth_equations[depth_equation][0]
                discount = depth_equations[depth_equation][1]

            if event.key == pygame.K_p:
                if last_board and movie_moves :
                    play_moves(last_board, movie_moves, depth)

            if event.key == pygame.K_s:
                print("Saving game...", board)
                #just draw white where the text will be
                pygame.draw.rect(screen, WHITE, (27, SCREEN_HEIGHT - 75, 200, 50))  
                screen.blit(font_info.render("Saving game...", True, BLACK), (27, SCREEN_HEIGHT - 75))
                read_aloud("Saving game")
                save_game(board, move_number, player, ai, depth, evaluation_method, ai_method_white, ai_method_black, depth_equation, show_simulation, list_of_boards, position_history, has_moved_history, game_history, game_history_simple)
            
            if event.key == pygame.K_l:
                old_board = board
                old_height = SCREEN_HEIGHT
                info = loop_to_select_new_game(screen, board)
                SCREEN_HEIGHT = old_height  
                #reset pygame display
                pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                #make the screen white
                pygame.draw.rect(screen, WHITE, (0,0,SCREEN_WIDTH,SCREEN_HEIGHT))
                if not info:
                    board = old_board
                    print("No game loaded.")
                    #just draw white where the text will be
                    pygame.draw.rect(screen, WHITE, (27, SCREEN_HEIGHT - 75, 200, 50))      
                    screen.blit(font_info.render("No game loaded.", True, BLACK), (27, SCREEN_HEIGHT - 75))
                    read_aloud("No game loaded")
                else:
                    try:
                        board = info.get("board", board)
                        move_number = info.get("move_number", move_number)
                        player = info.get("player", player)
                        ai = info.get("ai", ai)
                        depth = info.get("depth", depth)

                        evaluation_method = info.get("evaluation_method", evaluation_method)
                        evaluate_board = evaluation_methods.get(evaluation_method, evaluate_board) # important to reset!

                        ai_method_white = info.get("ai_method_white", ai_method_white)
                        ai_method_black = info.get("ai_method_black", ai_method_black)

                        depth_equation = info.get("depth_equation", depth_equation)
                        depth_formula = depth_equations.get(depth_equation, (depth_formula, discount))[0] # important to reset!
                        discount = depth_equations.get(depth_equation, (depth_formula, discount))[1]   # important to reset!

                        show_simulation = info.get("show_simulation", show_simulation)
                        list_of_boards = info.get("list_of_boards", list_of_boards)
                        position_history = info.get("position_history", position_history)
                        game_history_simple = info.get("game_history_simple", game_history_simple)
                        game_history = info.get("game_history", game_history)
                        has_moved_history = info.get("has_moved_history", has_moved_history)
                        has_moved = has_moved_history[-1]
                    except:
                        print("Error loading game.")
                        #just draw white where the text will be
                        pygame.draw.rect(screen, WHITE, (27, SCREEN_HEIGHT - 75, 200, 50))  
                        screen.blit(font_info.render("Error loading game.", True, BLACK), (27, SCREEN_HEIGHT - 75))
                        read_aloud("Error loading game")
                    print("Loaded game...", board)
                    pygame.draw.rect(screen, WHITE, (27, SCREEN_HEIGHT - 75, 200, 50))    
                    screen.blit(font_info.render("Loaded game...", True, BLACK), (27, SCREEN_HEIGHT - 75))
                    read_aloud("Loaded game")
                pygame.display.set_caption("JMR's Game of Chess Game")
                    
            #auto switch colors for player each time
            if event.key == pygame.K_v:
                setauto_switch_colors_for_player = not setauto_switch_colors_for_player
                if setauto_switch_colors_for_player:
                    print("Self play is now on")
                    read_aloud("Self play is now on")
                else:
                    print("Self play is now off")
                    read_aloud("Self play is now off")

            #Switching colors for player
            if event.key == pygame.K_x:
                if player == "W":
                    player = "B"
                    ai = "W"
                    player_turn = False
                   
                else:
                    player = "W"
                    ai = "B"
                    player_turn = False
                
                print(f"Player is now {piece_dict [player]}.")
                read_aloud(f"Player is now {piece_dict[player]}.")
                print(f"AI is now {piece_dict[ai]}.")
                read_aloud(f"AI is now {piece_dict[ai]}.")

            if event.key == pygame.K_b:  # 'b' for board reverse
                board_reversed = not board_reversed
                print(f"Board view reversed: {'Black' if board_reversed else 'White'} on bottom")
                read_aloud(f"Board view reversed: {'Black' if board_reversed else 'White'} on bottom")  
                pygame.draw.rect(screen, WHITE, (0,0,SCREEN_WIDTH,SCREEN_HEIGHT -200))
                draw_board_wrapper(screen, board)
                draw_pieces_not_on_board(screen, board, height=SCREEN_HEIGHT)
                pygame.display.flip()

            if event.key == pygame.K_2:
                display_mode = '2D'
                SCREEN_HEIGHT = 1400
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                #make the screen white
                pygame.draw.rect(screen, WHITE, (0,0,SCREEN_WIDTH,SCREEN_HEIGHT))
                print("Switching to 2D display mode.")
                read_aloud("Switching to 2D display mode.")
                pygame.draw.rect(screen, WHITE, (0,0,SCREEN_WIDTH,SCREEN_HEIGHT - 200)) 
                draw_board_wrapper(screen, board)
                pygame.display.flip()

            if event.key == pygame.K_3:
                display_mode = '3D'
                SCREEN_HEIGHT = 1000
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                #make the screen white
                pygame.draw.rect(screen, WHITE, (0,0,SCREEN_WIDTH,SCREEN_HEIGHT))
                print("Switching to 3D display mode.")
                read_aloud("Switching to 3D display mode.")
                pygame.draw.rect(screen, WHITE, (0,0,SCREEN_WIDTH,SCREEN_HEIGHT - 200)) 
                draw_board_wrapper(screen, board)
                pygame.display.flip()

            if event.key == pygame.K_r:
                initialize_game()
            if event.key == pygame.K_h:
                help()

            pygame.time.wait(100)
            draw_board_wrapper(screen, board)
            draw_pieces_not_on_board(screen, board, height=SCREEN_HEIGHT)
            pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 150, SCREEN_WIDTH - 50,50))
            screen.blit(font_info.render(f"Move: {move_number}. Player: {player}.  Ai white: {ai_method_white}. Ai black: {ai_method_black}. Depth equation: {depth_equation}", True, BLACK), (27, SCREEN_HEIGHT - 150))
            screen.blit(font_info.render(f"Depth: {depth}. Evaluation: {evaluation_method}. Simulation: {'On' if show_simulation else 'Off'}", True, BLACK), (27, SCREEN_HEIGHT - 125))

            pygame.display.flip()
         
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if player_turn and not end_of_game:
                moves = get_all_legal_moves(board, player, last_move = actual_last_move)
                if moves:
                    pos = screen_to_board_position(event.pos)
                    if pos is not None and pos[0] < BOARD_SIZE and pos[1] < BOARD_SIZE:
                        if selected_piece:
                            move = (selected_piece, pos)
                            if move in moves:
                                # Clear all saved history if we're making a new move
                                # after having gone back in history
                                if saved_game_history is not None:
                                    print("Making new move - clearing saved history")
                                    saved_game_history = None
                                    saved_game_history_simple = None
                                    saved_list_of_boards = None
                                    saved_position_history = None
                                    saved_has_moved_history = None
                                    
                                    # Also truncate current histories to current position
                                    list_of_boards = list_of_boards[:move_number + 1]
                                    game_history = game_history[:move_number * 2 + 1]
                                    game_history_simple = game_history_simple[:move_number * 2 + 1]
                                    position_history = position_history[:move_number * 2 + 1]
                                    has_moved_history = has_moved_history[:move_number + 1]

                                # Process the move
                                to_notation = f"{chr(97 + pos[1])}{8 - pos[0]}"
                                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 25, SCREEN_WIDTH - 50, 25))
                                screen.blit(font_info.render(f"From: {from_notation} To: {to_notation}", True, BLACK), (27, SCREEN_HEIGHT - 25))
                                pygame.display.flip()
                                
                                notation = convert_to_standard_notation(board, move)
                                game_history.append(notation)
                                notation_simple = convert_to_standard_notation_simple(move)
                                game_history_simple.append(notation_simple) 
                                piece = board[selected_piece[0]][selected_piece[1]]
                                print(f"Player moves: {piece} {move}")
                                print(f"Move made - move_number: {move_number}, boards: {len(list_of_boards)}, history: {len(game_history)}")
                                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 100, SCREEN_WIDTH - 50,50))
                                
                                screen.blit(font_info.render("Player moves: "+notation, True, BLACK), (27, SCREEN_HEIGHT - 100))
                                board = simulate_move(board, move, real_board=True)
                                position_history.append(board_to_hashable(board, ai))  # AI moves next
                                board_hash = board_to_hashable(board, ai)
                                actual_last_move = move #Track the last move
                                if piece in has_moved:
                                    has_moved[piece] = True
                                
                                draw_board_wrapper(screen, board)
                                draw_pieces_not_on_board(screen, board, height=SCREEN_HEIGHT)
                                
                                pygame.display.flip()
                                read_aloud("Player moves "+notation)
                                player_turn = False
                                selected_piece = None  # Reset selected piece after move
                                
                                # Check for checkmate and other game-ending conditions here
                                if is_checkmate(board, ai):
                                    print(f"Checkmate. {piece_dict[player]} wins.")
                                    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50, 50))
                                    screen.blit(font.render(f"Checkmate. {piece_dict[player]} wins.", True, BLACK), (27, SCREEN_HEIGHT - 50))
                                    read_aloud(f"Checkmate. {piece_dict[player]} wins.")
                                    end_of_game = True
                                    pygame.display.flip() 
                                elif can_claim_draw():
                                    print("Threefold repetition. Player can claim a draw.")
                                    pygame.display.set_caption("Threefold repetition.'D' draw or any key to continue.")
                                    pygame.display.flip()
                                    waiting_for_draw_decision = True
                                    while waiting_for_draw_decision:
                                        for event in pygame.event.get():
                                            if event.type == pygame.KEYDOWN:
                                                if event.key == pygame.K_d:
                                                    print("Draw claimed. Game over.")
                                                    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50, 50))
                                                    screen.blit(font.render("Draw claimed. Game over.", True, BLACK), (27, SCREEN_HEIGHT - 50))
                                                    read_aloud("Draw claimed. Game over.")
                                                    end_of_game = True
                                                else:
                                                    pygame.display.set_caption("JMR's Game of Chess")
                                                waiting_for_draw_decision = False
                                    if end_of_game:
                                        continue  # Skip to next iteration of main game loop
                            else:
                                print("Illegal move")
                                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))
                                screen.blit(font_info.render("Illegal move", True, BLACK), (27, SCREEN_HEIGHT - 50))
                                selected_piece = None
                        else:
                            if board[pos[0]][pos[1]] and board[pos[0]][pos[1]].startswith(player):
                                selected_piece = pos
                                from_notation = f"{chr(97 + pos[1])}{8 - pos[0]}"
                                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 25, SCREEN_WIDTH - 50, 25))
                                screen.blit(font_info.render(f"From: {from_notation}", True, BLACK), (27, SCREEN_HEIGHT - 25))
                                pygame.display.flip()
                            else:
                                selected_piece = None
                                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 25, SCREEN_WIDTH - 50, 25))
                                screen.blit(font_info.render("Select a valid piece", True, BLACK), (27, SCREEN_HEIGHT - 25))
                                pygame.display.flip()
                    else:
                        print("Clicked outside the board")
                        pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))
                        screen.blit(font_info.render("Clicked outside the board", True, BLACK), (27, SCREEN_HEIGHT - 50))
                else:
                    print("No legal moves for Player")
                    print("Stalemate.")
                    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,200))
                    screen.blit(font.render(f"No legal moves for {piece_dict[player]} Player  ", True, BLACK), (27, SCREEN_HEIGHT - 200))
                    screen.blit(font.render("Stalemate.", True, BLACK), (27, SCREEN_HEIGHT - 150))
                    pygame.display.flip()
                    read_aloud(f"No legal moves for {piece_dict[player]} AI  ")
                    read_aloud("Stalemate.")
                    end_of_game = True

            # Update display after each click
            #pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 150, SCREEN_WIDTH - 50,50))
            #screen.blit(font_info.render(f"Move: {move_number}. Player: {player}.  Ai: {ai_method}. {depth_equation}", True, BLACK), (27, SCREEN_HEIGHT - 150))
            #screen.blit(font_info.render(f"Depth: {depth}. Evaluation: {evaluation_method}. Show simulation: {show_simulation}", True, BLACK), (27, SCREEN_HEIGHT - 125))
            pygame.display.flip()

    # AI Turn Processing Section
    # In self-play mode, both sides are AI - alternate between White and Black
    if setauto_switch_colors_for_player and not end_of_game:
        # Alternate between White and Black AI turns
        # Start with White for first move
        if move_number == 0:
            ai = "W"
            ai_method = ai_method_white
        elif ai == "W":
            ai = "B"
            ai_method = ai_method_black
        else:
            ai = "W"
            ai_method = ai_method_white
        player_turn = False  # Always AI's turn in self-play
        # Skip read_aloud in self-play to avoid blocking on macOS 'say' command

    if not player_turn and not end_of_game:
        # Adding a timer to the AI's move
        start_time = time.time()
        ai_depth = depth  # You can ask the user for this input or adjust as needed
        move_number = move_number + 1

        # Set ai_method for non-self-play mode (player vs AI)
        if not setauto_switch_colors_for_player:
            # For player vs AI mode, AI is the opponent
            ai = "B" if player == "W" else "W"
            ai_method = ai_method_black if player == "W" else ai_method_white

        # ai_method is already set above for self-play
        print(f"AI {piece_dict[ai]} {ai_method} is thinking...")
        pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,100))
        screen.blit(font.render(f"AI {piece_dict[ai]} {ai_method} is thinking...", True, BLACK), (27, SCREEN_HEIGHT - 200)) 
        screen.blit(font_info.render(f"Move: {move_number}. Player: {player}.  Ai white: {ai_method_white}. Ai black: {ai_method_black}. Depth equation: {depth_equation}", True, BLACK), (27, SCREEN_HEIGHT - 150))
        screen.blit(font_info.render(f"Depth: {depth}. Evaluation: {evaluation_method}. Simulation: {'On' if show_simulation else 'Off'}", True, BLACK), (27, SCREEN_HEIGHT - 125))
        pygame.display.flip()
        # Clear transposition table between moves to prevent accumulation of irrelevant positions
        print(f"Transposition table size before clearing: {len(transposition_table)}")
        transposition_table.clear()
        print("Cleared transposition table for fresh start")
        # In self-play with same AI model, disable global optimizations
        # to prevent cross-contamination between identical AI instances
        in_self_play_identical = setauto_switch_colors_for_player and ai_method_white == ai_method_black == "Improved"
        if hasattr(select_best_ai_move_improved, '_in_self_play'):
            delattr(select_best_ai_move_improved, '_in_self_play')
        if in_self_play_identical:
            # Mark the AI function to disable global optimizations
            select_best_ai_move_improved._in_self_play = True

        pygame.event.pump() # so the AI can think without being interrupted by other events

        start_time = time.time()
        result = ai_methods[ai_method](board, ai_depth, ai, ai, float('-inf'), float('inf'), show_simulation, actual_last_move, ai_depth)
        end_time = time.time()
        if len(result) == 3:
            eval_score, selected_move, optionalisitLLMmove = result
        else:
            eval_score, selected_move = result
            optionalisitLLMmove = None

        #print("The path to get here: ", selected_move)
        # Debug: check if AI returned empty/invalid moves
        if selected_move is None or (isinstance(selected_move, list) and len(selected_move) == 0):
            print(f"DEBUG: AI {ai_method} returned empty moves for {ai}. Board state may be corrupted.")
            print(f"DEBUG: Last move was {actual_last_move}, current player to move: {ai}")

            # Determine if it's checkmate or stalemate
            if is_in_check(board, ai):
                # Checkmate - player wins
                print(f"Checkmate. {piece_dict[player]} player wins.")
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))
                screen.blit(font.render(f"Checkmate. {piece_dict[player]} player wins.", True, BLACK), (27, SCREEN_HEIGHT - 50))
                read_aloud(f"Checkmate. {piece_dict[player]} player wins.")
                pygame.display.flip()
            else:
                # Stalemate - draw
                print("Stalemate. Game is a draw.")
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))
                screen.blit(font.render("Stalemate. Game is a draw.", True, BLACK), (27, SCREEN_HEIGHT - 50))
                read_aloud("Stalemate. Game is a draw.")
                pygame.display.flip()

            # Force end game to prevent infinite loop
            end_of_game = True
            continue

        movie_moves = selected_move

        if selected_move and (not isinstance(selected_move, list) or len(selected_move) > 0):
            selected_move = movie_moves[0] if isinstance(movie_moves, list) else movie_moves
            notation = convert_to_standard_notation(board, selected_move)
            game_history.append(notation)
            notation_simple = convert_to_standard_notation_simple(selected_move)
            game_history_simple.append(notation_simple)
            new_position = selected_move[0]  #before the move you need to do the from :)
            piece = board[new_position[0]][new_position[1]]
            last_board = copy.deepcopy(board) # for movie playback reducant with list of boards but good for now
            board = simulate_move(board, selected_move, real_board=True)
            position_history.append(board_to_hashable(board, player))  # Player moves next
            board_hash = board_to_hashable(board, player)
            end_time = time.time()
            print(f"AI selects move: {piece} {selected_move}, eval score: {eval_score:.2f}, time: {end_time - start_time:.2f} seconds, moves considered: {len(transposition_table)}, paths: {len(movie_moves)}")
            print(f"Move made - move_number: {move_number}, boards: {len(list_of_boards)}, history: {len(game_history)}")
            pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 75, SCREEN_WIDTH - 50,25))
            screen.blit(font_info.render(f"AI: {notation}, {optionalisitLLMmove}, Ev: {eval_score:.2f}, Moves considered: {len(transposition_table)}, Time: {end_time - start_time:.0f}, Path: {len(movie_moves)} ", True, BLACK), (27, SCREEN_HEIGHT - 75))
            actual_last_move = selected_move #Track the last move added to use one call for both making and simulating.
            draw_board_wrapper(screen, board)
            draw_pieces_not_on_board(screen, board, height=SCREEN_HEIGHT)
            pygame.display.flip()
            # Skip read_aloud in self-play to avoid blocking on macOS 'say' command
            if not setauto_switch_colors_for_player:
                read_aloud("AI moves " + notation)
                read_aloud("your turn")
        else:
            # AI couldn't find any moves - this shouldn't happen in normal play
            print(f"AI {piece_dict[ai]} found no valid moves!")
            pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 75, SCREEN_WIDTH - 50,25))
            screen.blit(font_info.render(f"AI {piece_dict[ai]}: No moves found!", True, BLACK), (27, SCREEN_HEIGHT - 75))
            pygame.display.flip()
            end_of_game = True
            continue

        if selected_move and (not isinstance(selected_move, list) or len(selected_move) > 0):
            list_of_boards = list_of_boards[:move_number+1]
            list_of_boards.append(copy.deepcopy(board))
            has_moved_history.append(copy.deepcopy(has_moved))
            # Only set player_turn=True if not in self-play mode
            if not setauto_switch_colors_for_player:
                player_turn = True
            if is_in_check(board, player):
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,50))
                print(f"{piece_dict[player]} player  is in check.")
                screen.blit(font.render(f"{player} is in check.", True, BLACK), (27, SCREEN_HEIGHT - 200))
                if not setauto_switch_colors_for_player:
                    read_aloud(f"{piece_dict[player]} is in check.")
                pygame.display.flip()
            if is_checkmate(board, player):
                print(f"Checkmate. AI {piece_dict[ai]} wins.")
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))    
                screen.blit(font.render(f"Checkmate. {piece_dict[ai]} AI wins.", True, BLACK), (27, SCREEN_HEIGHT - 50))
                if not setauto_switch_colors_for_player:
                    read_aloud(f"Checkmate. {piece_dict[ai]} AI wins.")
                end_of_game = True
                pygame.display.flip() 
            if is_automatic_draw():
                print("Fivefold repetition. Automatic draw.")
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))    
                screen.blit(font.render("Fivefold repetition. Game is a draw.", True, BLACK), (27, SCREEN_HEIGHT - 50))
                if not setauto_switch_colors_for_player:
                    read_aloud("Fivefold repetition. Game is a draw.")
                end_of_game = True
                pygame.display.flip()
            elif can_claim_draw():
                print ("Can claim draw")
                if ai_should_claim_draw(board, ai):
                    print("Threefold repetition. AI claims draw.")
                    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50, 50))    
                    #specify the color of the AI
                    screen.blit(font.render(f"Threefold repetition. {piece_dict[ai]} AI claims draw.", True, BLACK), (27, SCREEN_HEIGHT - 50))
                    if not setauto_switch_colors_for_player:
                        read_aloud("Threefold repetition. AI claims draw. Game over.")
                    end_of_game = True
                    pygame.display.flip()
                else:
                    print("AI chooses not to claim draw.")
                    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50, 50))    
                    screen.blit(font.render("AI chooses not to claim draw.", True, BLACK), (27, SCREEN_HEIGHT - 50))
                    if not setauto_switch_colors_for_player:
                        read_aloud("AI chooses not to claim draw.")
                    pygame.display.flip()

        else:
            print("No legal moves for AI")
            print("Stalemate.")
            pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,200))
            screen.blit(font.render(f"No legal moves for {piece_dict[ai]} AI  ", True, BLACK), (27, SCREEN_HEIGHT - 200))
            screen.blit(font.render("Stalemate.", True, BLACK), (27, SCREEN_HEIGHT - 150))
            pygame.display.flip()
            read_aloud(f"No legal moves for {piece_dict[ai]} AI  ")
            read_aloud("Stalemate.")
            end_of_game = True

    if end_of_game:
        if not auto_save:
            print("Saving game...", board)
            pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 75, SCREEN_WIDTH - 50,30))
            screen.blit(font_info.render("Saving game...", True, BLACK), (27, SCREEN_HEIGHT - 75))
            read_aloud("Saving game")
            save_game(board, move_number, player, ai, depth, evaluation_method, ai_method_white, ai_method_black, depth_equation, show_simulation, list_of_boards, position_history, has_moved_history, game_history, game_history_simple)

            # Display LLM move statistics
            if llm_stats['total_moves'] > 0:
                print("\n=== LLM Move Statistics ===")
                print(f"Total LLM moves: {llm_stats['total_moves']}")
                print(f"First response legal: {llm_stats['first_legal']} times")
                print(f"Second response legal: {llm_stats['second_legal']} times")
                print(f"Third+ response legal: {llm_stats['third_plus_legal']} times")
                print(f"No legal moves found: {llm_stats['no_legal']} times")

            read_aloud("Hit 'r' to restart the game.")
            pygame.display.set_caption("Hit 'r' to restart the game.")
            pygame.display.flip()
            auto_save = True
            player_turn = True
pygame.quit()
sys.exit()