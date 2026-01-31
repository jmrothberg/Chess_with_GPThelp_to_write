#JMR Simple Chess Game March 4 2024
#Removed large number of the AI methods and board evaluatoins to keep code base smaller for LLM help
#Added LLM to game Sept 19
#Added simple notation for internal use and use of chess move tokenizer Sept 24, promotes to knigth if checkmate
#Added top k responses from LLM to game and 3 and 5 move repetition for draw   Sept 25
#Changed to CPU Sept 26 and also made sure no crash when no moves are found but does stalemate.
#Allow board to be rotated 180 degrees. 3D board added Sept 27
#Cleaned up save game Sept 29, fallback is best evaluation
#Draw logic corrected Sept 30
#Added ability to have different AI for each color Oct 5
#Added quiescence search Oct 9 for best_improved
#oct 15, bug was clearing the screen for 2D board, and need to just do for 3D board

import sys
import pygame
import copy
import time
import os
import string
import json
import BrainInference_Oct_5_VIS as brain_inference
from BrainInference_Oct_5_VIS import generate_response
from game_selector3D import loop_to_select_new_game, draw_pieces_not_on_board
import random
import platform
import math

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
# Chess_Sept_27_LMM_3D.py
OFF_WHITE = (200, 200, 200)  # Slightly darker than pure white

sound_enabled = True
board_reversed = False
display_mode = '2D'

# Initialize the screen with given dimensions
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
pygame.display.set_caption("JMR's Game of Chess Game: Press H for Help")
font = pygame.font.SysFont("Arial", 42)
font_moves = pygame.font.SysFont("Arial", 33)
font_info = pygame.font.SysFont("Arial", 24)
font_big = pygame.font.Font("ARIALUNI.TTF", 87)
font_2big = pygame.font.Font("ARIALUNI.TTF", 120)

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
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = board[r][c]
            if piece.startswith(attacker_color):
                moves = get_moves_for_piece(board, r, c, last_move=None, check_castling=False)
                if any(end == (row, col) for _, end in moves):
                    return True
    return False


# Moved board back but without using simulation since we know all the moves are legal. just need to check if the king is in check
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

# Function to check for checkmate
def is_checkmate(board, color):
    if not get_all_legal_moves(board, color) and is_in_check(board, color):
        return True
    return False

#Need to use for evaluaton function when ahead on material so don't get stuck in a draw
def is_stalemate(board, color):
    return not get_all_legal_moves(board, color) and not is_in_check(board, color)

def board_to_hashable(board, player_to_move): #need to make board hashable for transposition table for stalemate
    return (tuple(tuple(row) for row in board), player_to_move)

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
    captured_piece = board[end[0]][end[1]]

    # Make a shallow copy of the board
    new_board = [row[:] for row in board]

    if real_board or show_simulation:
        notation = convert_to_chess_notation(new_board, move)

    if real_board:
        pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,50))
        pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))
    
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
            
            if show_simulation:
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
            if show_simulation:
                print(f"Simulated en passant capture. {notation}.")
            if real_board:
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 150, SCREEN_WIDTH - 50,50))
                screen.blit(font.render(f"en passant capture. {notation}", True, BLACK), (27 ,SCREEN_HEIGHT - 150))
                read_aloud(f"en passant capture. {notation}")
        else:
            new_board[end[0]-1][end[1]] = ""
            if show_simulation:
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
            if show_simulation:
                print(f"Simulated {piece_dict[color]} castling.")
            if real_board:
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 195, SCREEN_WIDTH - 50,50))
                screen.blit(font_info.render(f"{piece_dict[color]} king-side castling.", True, BLACK), (27, SCREEN_HEIGHT - 195))
                read_aloud(f"{piece_dict[color]} king-side castling.")
        elif end[1] == start[1]-2:  # Queen-side castling
            new_board[start[0]][start[1]-1] = color + 'R1'
            new_board[start[0]][start[1]-4] = ''
            if show_simulation:
                print(f"Simulated {piece_dict[color]} castling.")   
            if real_board:
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 195, SCREEN_WIDTH - 50, 50))
                screen.blit(font_info.render(f"{piece_dict[color]} queen-side castling.", True, BLACK), (27, SCREEN_HEIGHT - 195))
                read_aloud(f"{piece_dict[color]} queen-side castling.")

    # Capture logic 
    if captured_piece != "":
        if show_simulation:
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


def select_best_ai_move_improved(board, depth, color, AI_color, alpha=float('-inf'), beta=float('inf'), display_simulation=False, last_move=None, initial_depth=None):
    global transposition_table, piece_values, depth_formula, discount
    
    board_key = tuple(map(tuple, board))
    if board_key in transposition_table:
        eval_board, eval_depth, best_move_new = transposition_table[board_key]
        if display_simulation:
           print(f"Transposition Table hit for {color} {depth} {eval_board:.2f} {eval_depth} {best_move_new} {last_move}")
        if eval_depth >= depth:
            return eval_board, best_move_new

    legal_moves = get_all_legal_moves(board, color, last_move=last_move, check_legality=True)
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
        return quiescence_search(board, color, AI_color, alpha, beta, 0, 3)

    # Null Move Pruning
    if depth > 2 and not check:
        null_move_eval, _ = select_best_ai_move_improved(board, depth - 3, 'B' if color == 'W' else 'W', AI_color, -beta, -beta + 1, display_simulation, last_move, initial_depth)
        null_move_eval = -null_move_eval
        if null_move_eval >= beta:
            return beta, []

    # Move Ordering
    def order_moves(board, moves, color):
        ordered_moves = []
        for move in moves:
            score = 0
            piece = board[move[0][0]][move[0][1]]
            target = board[move[1][0]][move[1][1]]
            
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

    best_eval = float('-inf') if color == AI_color else float('inf')
    best_move_new = []

    for move_index, (move, _) in enumerate(scored_moves):
        new_board = simulate_move(board, move, real_board=False)

        # Late Move Reduction
        if depth >= 3 and move_index > 3 and not check and not board[move[1][0]][move[1][1]]:
            eval_board, opponent_best_move = select_best_ai_move_improved(new_board, depth-2, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            if (color == AI_color and eval_board < alpha) or (color != AI_color and eval_board > beta):
                continue

        eval_board, opponent_best_move = select_best_ai_move_improved(new_board, depth-1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
        
        if (color == AI_color and eval_board > best_eval) or (color != AI_color and eval_board < best_eval):
            best_eval = eval_board
            best_move_new = [move] + opponent_best_move if opponent_best_move else [move]

        if color == AI_color:
            alpha = max(alpha, best_eval)
        else:
            beta = min(beta, best_eval)
        
        if beta <= alpha:
            break

    transposition_table[board_key] = best_eval, depth, best_move_new
    return best_eval, best_move_new

def quiescence_search(board, color, AI_color, alpha, beta, depth, max_depth):
    stand_pat = evaluate_board(board)
    if depth >= max_depth:
        return stand_pat, []

    if color == 'W':
        if stand_pat >= beta:
            return beta, []
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return alpha, []
        beta = min(beta, stand_pat)
    
    captures = [move for move in get_all_legal_moves(board, color, check_legality=True) if board[move[1][0]][move[1][1]] != '']
    best_move = []
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
    
    return (alpha, best_move) if color == 'W' else (beta, best_move)


def select_best_ai_move_sort_equation(board, depth, color, AI_color, alpha=float('-inf'), beta=float('inf'), display_simulation=False, last_move=None, initial_depth=None):
    global transposition_table
    
    board_key = tuple(map(tuple, board))
    if board_key in transposition_table:
        eval_board,eval_depth, best_move_new = transposition_table[board_key]
        if display_simulation:
           print(f"Transposition Table hit for {color} {depth} {eval_board:.2f} {eval_depth} {best_move_new} {last_move}")
        if eval_depth >= depth:
            return eval_board, best_move_new
    board_table = {}

    # Initialize the list of scored moves
    scored_moves = []

    #Checks if MOVING gets you into check and is none if it doesn't get you out of check
    legal_moves = get_all_legal_moves(board, color, last_move=last_move,check_legality=True) #if you limit to legal you miss taking the king when legal moves are none!
    
    #Are you in check?
    check = is_in_check(board, color) # get all opponents moves and see if any of them end in your king's square

    evaluation = evaluate_board(board)

    #Avoid stalemate, is in check gets oppenents legal moves
    if not check and not legal_moves:
        evaluation = evaluation * (discount**((initial_depth +1 )- depth))
        if display_simulation:
            print(f"Stalemate, no legal moves for {color}. Evaluation: {-evaluation:.2f}")
            print (f"Depth: {depth}, Last move: {last_move}")
        return -evaluation, []  # Return empty list instead of string message

    # Checkmate if in check and no legal moves  #since you never really take the king when legal moves are none!
    if check and not legal_moves:
        if color == "W":
            evaluation = (evaluation  - 100) * (discount**((initial_depth+1) - depth))
        else:
            evaluation = (evaluation + 100) * (discount**((initial_depth +1 ) - depth))
        if display_simulation:
            print(f"Checkmate of {color}. Evaluation: {evaluation:.2f}")
            print (f"Depth: {depth}, Last move: {last_move} of {'w' if color == 'B' else 'B'}")
        return evaluation, []  # Return empty list instead of string message
    
    if depth == 0:
        evaluation = evaluation * (discount**((initial_depth +1) - depth))
        return evaluation, [f"End depth {color} {depth}"]

    for move in legal_moves:
        new_board = simulate_move(board, move, real_board=False)
        board_table[(tuple(map(tuple, board)), move)] = new_board
        score = evaluate_board(new_board) 

        scored_moves.append((move, score))

    if color == 'W':
        scored_moves.sort(key=lambda x: x[1], reverse=True)  # White maximizes
    else:
        scored_moves.sort(key=lambda x: x[1])  # Black minimizes

    length = len(scored_moves)
    if length > eval(depth_formula):
        scored_moves = scored_moves[:eval(depth_formula)]

    if AI_color == 'B':
        
        best_eval = float('inf') if color == AI_color else float('-inf')
        best_move_new = []
    
        for move, _ in scored_moves:
           
            move_key = (tuple(map(tuple, board)), move)
            if move_key in board_table:
                new_board = board_table[move_key]
            
            eval_board, opponent_best_move = select_best_ai_move_sort_equation(new_board, depth-1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            
            if (color == AI_color and eval_board < best_eval) or (color != AI_color and eval_board > best_eval):
                best_eval = eval_board
                best_move_new = [move] + opponent_best_move if opponent_best_move else [move]

            transposition_table[tuple(map(tuple, new_board))] = best_eval, depth, best_move_new

            if color == AI_color:
                beta = min(beta, best_eval) # when its a new low eval and it is AI turn (this makes sense since black is minimizing)
            else:
                alpha = max(alpha, best_eval) # when AI is black, but its white turns and it is maximizing
            if beta <= alpha:
                #print(f"Break beta <= alpha, Depth: {depth}, Color: {color}, after alpha return Best move: {best_move}, value: {best_eval}")
                break

    if AI_color == 'W':
        
        best_eval = float('-inf') if color == AI_color else float('inf')
        best_move_new = []
        
        for move, _ in scored_moves:
            
            move_key = (tuple(map(tuple, board)), move)
            if move_key in board_table:
                new_board = board_table[move_key]
            
            eval_board, opponent_best_move = select_best_ai_move_sort_equation(new_board, depth-1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            
            if (color == AI_color and eval_board > best_eval) or (color != AI_color and eval_board < best_eval):
                best_eval = eval_board
                best_move_new = [move] + opponent_best_move if opponent_best_move else [move]

            if color == AI_color:
                beta = max(beta, best_eval) 
            else:
                alpha = min(alpha, best_eval) 
            
            transposition_table[tuple(map(tuple, new_board))] = best_eval, depth, best_move_new

            if beta <= alpha:
                #print(f"Break beta <= alpha, Depth: {depth}, Color: {color}, after alpha return Best move: {best_move}, value: {best_eval}")
                break

    return best_eval, best_move_new


def select_best_ai_move_min_max(board, depth, color, AI_color, alpha=float('-inf'), beta=float('inf'), display_simulation=False, last_move=None, initial_depth=None):
    #Checks if MOVING gets you into check and is none if it doesn't get you out of check
    legal_moves = get_all_legal_moves(board, color, last_move=last_move,check_legality=True) #if you limit to legal you miss taking the king when legal moves are none!
    
    #Are you in check?
    check = is_in_check(board, color) # get all opponents moves and see if any of them end in your king's square

    evaluation = evaluate_board(board)

    # Checkmate if in check and no legal moves  #since you never really take the king when legal moves are none!
    if check and not legal_moves:
        if color == "W":
            evaluation = (evaluation  - 100) * (discount**((initial_depth+1) - depth))
        else:
            evaluation = (evaluation + 100) * (discount**((initial_depth +1 ) - depth))
        if display_simulation:
            print(f"Potential Checkmate of {color}. Evaluation: {evaluation:.2f}")
            print (f"Depth: {depth}, Last move: {last_move} of {'w' if color == 'B' else 'B'}")
        return evaluation, [f"End Checkmate {color} {depth}"]

    #Avoid stalemate, is in check gets oppenents legal moves
    if not check and not legal_moves:
        evaluation = evaluation * (discount**((initial_depth +1 )- depth))
        if display_simulation:
            print(f"Potential Stalemate, no legal moves for {color}. Evaluation: {-evaluation:.2f}")
            print (f"Depth: {depth}, Last move: {last_move}")
        return -evaluation, [f"End stalemate {color} {depth}"]
    
    if depth == 0:
        evaluation = evaluation * (discount**((initial_depth +1) - depth))
        return evaluation, [f"End depth {color} {depth}"]

    if AI_color == 'B':
        best_move_new = []
        best_eval = float('inf') if color == AI_color else float('-inf')
        
        for move in legal_moves:
            new_board = simulate_move(board, move)
           
            # Check if the opponent's king is in checkmate after the move
            if is_in_check(new_board, 'W') and not get_all_legal_moves(new_board, 'W', last_move=move, check_legality=True):
                return float('inf'), [(move, depth)]

            eval_board, opponent_best_move = select_best_ai_move_min_max(new_board, depth - 1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            
            if (color == AI_color and eval_board < best_eval) or (color != AI_color and eval_board > best_eval):
                best_eval = eval_board
                best_move_new = [move] + opponent_best_move if opponent_best_move else [move]
               

            if color == AI_color:
                beta = min(beta, best_eval)
            else:
                alpha = max(alpha, best_eval)

            if beta <= alpha:
                if display_simulation:
                    print(f"Break beta <= alpha, Depth: {depth}, Color: {color}, after alpha return Best move: {best_move_new}, value: {best_eval}")
                break

    if AI_color == 'W':
        best_move_new =[]
        best_eval = float('-inf') if color == AI_color else float('inf')
        for move in legal_moves:
            new_board = simulate_move(board, move)
            eval_board, opponent_best_move = select_best_ai_move_min_max(new_board, depth - 1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            
            if (color == AI_color and eval_board > best_eval) or (color != AI_color and eval_board < best_eval):
                best_eval = eval_board
                best_move_new = [move] + opponent_best_move if opponent_best_move else [move]
            

            if color == AI_color:
                beta = max(beta, best_eval)
            else:
                alpha = min(alpha, best_eval)
            if beta <= alpha:
                if display_simulation:
                    print(f"Break beta <= alpha, Depth: {depth}, Color: {color}, after alpha return Best move: {best_move_new}, value: {best_eval}")
                break
    return best_eval, best_move_new


def convert_to_standard_notation(board, move):
    """Convert a move from the game's format to standard chess notation."""
    start, end = move
    piece = board[start[0]][start[1]]
    piece_type = piece[1] if piece and len(piece) > 1 and piece[1] != 'P' else '' #updated 
    start_square = chr(97 + start[1]) + str(8 - start[0])
    end_square = chr(97 + end[1]) + str(8 - end[0])
    return f"{piece_type}{start_square}{end_square}"


def convert_to_standard_notation_simple(move):
    """Convert a move from the game's format to simple standard chess notation (just coordinates)."""
    start, end = move
    start_square = chr(97 + start[1]) + str(8 - start[0])
    end_square = chr(97 + end[1]) + str(8 - end[0])
    return f"{start_square}{end_square}"


def convert_from_standard_notation(board, standard_move):
    """Convert a move from standard chess notation to the game's format."""
    if len(standard_move) == 4:
        start_square, end_square = standard_move[:2], standard_move[2:]
    elif len(standard_move) == 5:
        start_square, end_square = standard_move[1:3], standard_move[3:]
    else:
        return None  # Invalid move format

    start_col, start_row = ord(start_square[0]) - 97, 8 - int(start_square[1])
    end_col, end_row = ord(end_square[0]) - 97, 8 - int(end_square[1])
    
    return ((start_row, start_col), (end_row, end_col))


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

        # Handle long algebraic notation (e.g., "e2e4" or "E2E4")
        if len(first_move) == 4 and first_move.isalnum():
            start_col, start_row = ord(first_move[0].lower()) - 97, 8 - int(first_move[1])
            end_col, end_row = ord(first_move[2].lower()) - 97, 8 - int(first_move[3])
            move = ((start_row, start_col), (end_row, end_col))
            #print(f"Long algebraic move: {move}")
            parsed_moves.append(move if not promotion else move + (promotion,))
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
    global llm_white, llm_black, block_size
    
    print(f"AI model initialized successfully for {color}.")
    if color == 'W':
        print("Initializing white model...")
        llm_white = brain_inference.initialize_model()
        if llm_white is None:
            print(f"Failed to initialize AI model for {color}. Exiting.")
            return False
        block_size = brain_inference.global_model.block_size # only allow one block size
    else:
        print("Initializing black model...")
        llm_black = brain_inference.initialize_model()
        if llm_black is None:
            print(f"Failed to initialize AI model for {color}. Exiting.")
            return False
        block_size = brain_inference.global_model.block_size

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
                new_board = simulate_move(board, first_move)
                evaluation = evaluate_board(new_board)
                return evaluation, suggested_moves, response_k+1  # Return all suggested moves for visualization
    
    # If no legal move is found in the top K responses, fall back to the best evaluated legal move
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
        return evaluation, [fallback_move] + suggested_moves[1:]  # Include fallback move and keep rest for visualization
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

# Evaluate the board for given state (basic evaluation without consideration of checkmate or check)
def evaluate_board_basic(board):
    
    piece_values = {
        'WP': 1, 'WN': 3, 'WB': 3, 'WR1': 5, 'WR2': 5, 'WQ': 9, 'WK': 100,
        'BP': -1, 'BN': -3, 'BB': -3, 'BR1': -5, 'BR2': -5, 'BQ': -9, 'BK': -100
    }

    evaluation = 0
    for row in board:
        for piece in row:
            if piece:
                value = piece_values[piece]
                # White pieces contribute positively to the score, black pieces negatively
                evaluation += value 
    return evaluation

def evaluate_board_positions_optimized(board):
   
    piece_values = {
        'WP': 1, 'WN': 3, 'WB': 3, 'e': 5, 'WR2': 5, 'WQ': 9, 'WK': 100,
        'BP': -1, 'BN': -3, 'BB': -3, 'BR1': -5, 'BR2': -5, 'BQ': -9, 'BK': -100
    }

    evaluation = 0

    get_piece_val = piece_values.get
    get_pos_val_white = {(piece, i, j): value for piece in positional_values.keys()
                         for i, row in enumerate(positional_values[piece])
                         for j, value in enumerate(row)}
    get_pos_val_black = {(piece, 7-i, j): -value for piece in positional_values.keys()
                         for i, row in enumerate(positional_values[piece])
                         for j, value in enumerate(row)}

    for i, row in enumerate(board):
        for j, piece in enumerate(row):
            if piece:
                is_white = piece[0] == 'W'
                piece_type = piece[1]
                value = get_piece_val(piece, 0)
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
            select_best_ai_move, ai_method, has_moved, auto_save, game_history, game_history_simple, position_history, board_reversed, sound_enabled, ai_method_white, ai_method_black, llm_white, llm_black, block_size, has_moved_history
    # The initial board setup, simplified without pawn promotion
    board = initial_board
    player_turn = True
    selected_piece = None
    actual_last_move = None  # Will hold the last move made in the game as a tuple: ((start_row, start_col), (end_row, end_col))
    list_of_boards = [copy.deepcopy(board)]
    move_number = 0
    player_turn = True
    selected_piece = None
    end_of_game = False
    running = True
    show_simulation = False
    end_of_game = False
    depth = 3
    transposition_table = {}
    player = "W"
    ai = "B"
    player_turn = True
    setauto_switch_colors_for_player = False
    auto_save = False

    # Add these global variables
    ai_method_white = "LLM"
    ai_method_black = "LLM"
    llm_white = None
    llm_black = None
    block_size = None
   
    sound_enabled = True
    board_reversed = False
    evaluation_method = "Positions_optimized"
    evaluate_board = evaluation_methods[evaluation_method]
    ai_method = "LLM"
    select_best_ai_move = ai_methods[ai_method]
    depth_equation = "All .95"
    depth_formula = depth_equations[depth_equation][0]
    discount = depth_equations[depth_equation][1]

    has_moved = {"WK": False, "WR1": False, "WR2": False, "BK": False, "BR1": False, "BR2": False}
    has_moved_history = [copy.deepcopy(has_moved)]

    game_history = ["<STARTGAME> "]
    game_history_simple = ["<STARTGAME>"]
    position_history = []  # List to store board positions and player to move
    screen.fill(WHITE)
    draw_board_wrapper(screen, board)
    draw_pieces_not_on_board(screen,board, height=SCREEN_HEIGHT)
    pygame.display.set_caption("JMR's Game of Chess")
    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,200))
    screen.blit(font_info.render(f"Move: {move_number}. Player: {player}.  Ai: {ai_method}. {depth_equation}", True, BLACK), (27, SCREEN_HEIGHT - 150))
    screen.blit(font_info.render(f"Depth: {depth}. Evaluation Method {evaluation_method}. Show simulation: {show_simulation}", True, BLACK), (27, SCREEN_HEIGHT - 125))
    pygame.display.flip()

    # Initialize the AI models
    print("Initializing LLM for White...")
    initialize_ai_model('W')
    print("Initializing LLM for Black...")
    initialize_ai_model('B')


help_page_1 = {
    "Move selection": "Use Mouse click to select move.",
    "Save/Load/Restart": "Press 's', 'l' or 'r' to save, load, restart.",
    "AI thinking display": "Press 'y' to toggle display of AI thinking.",
    "Speach": "Press 'n' to toggle speech.",
    "Set depth": "Press up-arrow or down-arrow to set depth.",
    "Select AI/evaluation": "Press 'a' or 'd' to select AI or equations for move selection.",
    "Evaluation mothods": "Press 'e' to cycle through evaluation methods.",
    "Roll back/forward": "Press left-arrow or right-arrow to roll back/forward moves.",
    "Switch sides": lambda: f"Player is {player} press 'x' to switch colors for player.",
    "Reverse board": "Press 'b' to reverse board view.",
    "Toggle self-play": "Press 'v' to toggle auto-switch colors (self-play mode).",
    "Replay AI moves": "Press 'p' to replay AI's considered moves.",
    "Help navigation": "Press 'h' to toggle help screen. Press 'space' for next page.",
    "Reload AI model": "Press 'm' to reload the AI model."
}

ai_methods = {
    'Sort_moves uses equations': select_best_ai_move_sort_equation,
    'Minimax': select_best_ai_move_min_max,
    'LLM': select_best_ai_move_llm,
    'Improved': select_best_ai_move_improved,  # Add this line
}

evaluation_methods = {
    'Basic': evaluate_board_basic,
    'Positions_optimized': evaluate_board_positions_optimized,
    }

evaluation_method_descriptions = {
    'Basic': "Simple material count. Fast but misses positional nuances.",
    'Positions_optimized': "Considers piece positions. Good balance of speed and accuracy.",
}

Ai_method_descriptions = {
    'Sort_moves uses equations': "Uses depth equations to dynamically reduce moves. More flexible pruning.",
    'Minimax': "Basic minimax algorithm. Doesn't use depth equations.",
    'LLM': "Custum JMR's LMM trained on chess games"
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
    page = 1
    font_help = pygame.font.SysFont("Arial", 14)  # Smaller font for the help screens
    while True:
        screen.fill(WHITE)
        draw_board_wrapper(screen, board)
        
        if page == 1:
            y = 605
            for key, value in help_page_1.items():
                if key == "Switch colors":
                    text = value()  # Call the lambda function
                else:
                    text = value
                screen.blit(font_help.render(f"{key}: {text}", True, BLACK), (20, y))
                y += 15
        elif page == 2:
            y = 605
            screen.blit(font_help.render("Search Methods:", True, BLACK), (20, y))
            y += 20
            for method, description in Ai_method_descriptions.items():
                screen.blit(font_help.render(f"{method}: {description}", True, BLACK), (20, y))
                y += 20
            screen.blit(font_help.render("Verse Mode:", True, BLACK), (20, y))
            y += 20
            screen.blit(font_help.render("Press 'v' to toggle Verse Mode", True, BLACK), (20, y))
            y += 20
            screen.blit(font_help.render("In Verse Mode:", True, BLACK), (20, y))
            y += 20
            screen.blit(font_help.render("Press 'a' to cycle White AI, 'z' to cycle Black AI", True, BLACK), (20, y))
            y += 20
            screen.blit(font_help.render("Press 'm' to initialize LLM for White or Black", True, BLACK), (20, y))
            y += 20
            screen.blit(font_help.render(f"Current White AI: {ai_method_white}", True, BLACK), (20, y))
            y += 20
            screen.blit(font_help.render(f"Current Black AI: {ai_method_black}", True, BLACK), (20, y))
            screen.blit(font_help.render("Press 'space' for next page, 'h' to exit help.", True, BLACK), (20, 780))
        elif page == 3:
            y = 605
            screen.blit(font_help.render("Evaluation Methods:", True, BLACK), (20, y))
            y += 25
            for method, description in evaluation_method_descriptions.items():
                screen.blit(font_help.render(f"{method}: {description}", True, BLACK), (20, y))
                y += 25
            screen.blit(font_help.render("Press 'space' for next page, 'h' to exit help.", True, BLACK), (20, 780))
        elif page == 4:
            y = 605
            screen.blit(font_help.render("Depth Equation:", True, BLACK), (20, y))
            screen.blit(font_help.render(f"{depth_equation}", True, BLACK), (130, y))
            screen.blit(font_help.render("Press 'd' to cycle through options:", True, BLACK), (260, y))
            y += 15
            screen.blit(font_help.render("Affects move reduction in 'smart_sort' and 'Sort_moves uses equations'.", True, BLACK), (20, y))
            y += 15
            for eq in depth_equations.keys():
                screen.blit(font_help.render(f"- {eq}", True, BLACK), (20, y))
                y += 15
            y += 10
            screen.blit(font_help.render("Iterative Deepening:", True, BLACK), (20, y))
            
            screen.blit(font_help.render(f"Currently {'enabled' if iterative_deepening_enabled else 'disabled'}", True, BLACK), (150, y))
            y += 15
            screen.blit(font_help.render("Press 'i' to toggle. Improves move ordering and search efficiency.", True, BLACK), (20, y))
            screen.blit(font_help.render("Press 'space' for first page, 'h' to exit help.", True, BLACK), (20, 780))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,200))
                    return
                elif event.key == pygame.K_SPACE:
                    page = (page % 4) + 1  # Cycle through pages 1, 2, 3, 4

# Add these dictionaries outside the help function:
player = "W"

# Global variable to store the wrapped AI method
wrapped_ai_method = None
iterative_deepening_enabled = False
# Main game loop
initialize_game()
selected_move = None
game_history_simple = ["<STARTGAME>"]   

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
                # Go to the previous board state if it exists
                if move_number > 0:
                    move_number -= 1
                    print("Going back to:", move_number)
                    board = copy.deepcopy(list_of_boards[move_number])
                    has_moved = copy.deepcopy(has_moved_history[move_number])
                    if current_branch_point is None:
                        current_branch_point = move_number + 1
                    if move_number < len(list_of_boards) - 1 and end_of_game:
                        end_of_game = False

            elif event.key == pygame.K_RIGHT:
                # Go to the next board state if it exists
                if move_number < len(list_of_boards) - 1:
                    move_number += 1
                    print("Going forward to: ", move_number)
                    board = copy.deepcopy(list_of_boards[move_number])
                    has_moved = copy.deepcopy(has_moved_history[move_number])
                    if move_number == current_branch_point:
                        current_branch_point = None
                    if move_number == len(list_of_boards) - 1 and end_of_game:
                        print("Reached end of game")
                    else:
                        end_of_game = False
                                

            #Changing evaluation method, moving through the dictionary, each time you press the key e key
            if event.key == pygame.K_e:
                keys = list(evaluation_methods.keys())
                index = keys.index(evaluation_method)
                if index < len(keys) - 1:
                    index += 1
                else:
                    index = 0
                evaluation_method = keys[index]
                evaluate_board = evaluation_methods[evaluation_method]
                print(f"Evaluation method is now {evaluation_method}.")

             #chaning AI method, moving through the dictionary, each time you press the key a key
            if event.key == pygame.K_a:
                keys = list(ai_methods.keys())
                if player == "W":
                    index = keys.index(ai_method_black)
                    if index < len(keys) - 1:
                        index += 1
                    else:
                        index = 0
                    ai_method_black = keys[index]
                    print(f"Black AI method is now {ai_method_black}.")
                else:
                    index = keys.index(ai_method_white)
                    if index < len(keys) - 1:
                        index += 1
                    else:
                        index = 0
                    ai_method_white = keys[index]
                    print(f"White AI method is now {ai_method_white}.")

            if event.key == pygame.K_z:
                keys = list(ai_methods.keys())
                index = keys.index(ai_method_black)
                if index < len(keys) - 1:
                    index += 1
                else:
                    index = 0
                ai_method_black = keys[index]
                print(f"Black AI method is now {ai_method_black}.")

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
                                 # Insert the branching logic here
                                if current_branch_point is not None and move_number < current_branch_point - 1:
                                    # We're creating a new branch
                                    list_of_boards = list_of_boards[:move_number + 1]
                                    game_history = game_history[:move_number + 1]
                                    game_history_simple = game_history_simple[:move_number + 1]
                                    position_history = position_history[:move_number + 1]
                                    has_moved_history = has_moved_history[:move_number + 1]
                                    current_branch_point = None

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

     # Check if it's AI's turn
    if setauto_switch_colors_for_player and not end_of_game:
        if player == "W":
                    player = "B"
                    ai = "W"
                    player_turn = False
        else:
            player = "W"
            ai = "B"
            player_turn = False
        screen.blit(font.render(f"Self_Play.", True, BLACK), (27, SCREEN_HEIGHT - 200))  
        print(f"Player is now {piece_dict [player]}.")
        read_aloud(f"Player is now {piece_dict[player]}.")
        print(f"AI is now {piece_dict[ai]}.")
        read_aloud(f"AI is now {piece_dict[ai]}.")
        
    if not player_turn and not end_of_game:
        # Adding a timer to the AI's move
        start_time = time.time()
        ai_depth = depth  # You can ask the user for this input or adjust as needed
        move_number = move_number + 1
        ai_method = ai_method_white if ai == 'W' else ai_method_black
        print(f"AI {piece_dict[ai]} {ai_method} is thinking...")
        pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,100))
        screen.blit(font.render(f"AI {piece_dict[ai]} {ai_method} is thinking...", True, BLACK), (27, SCREEN_HEIGHT - 200)) 
        screen.blit(font_info.render(f"Move: {move_number}. Player: {player}.  Ai white: {ai_method_white}. Ai black: {ai_method_black}. Depth equation: {depth_equation}", True, BLACK), (27, SCREEN_HEIGHT - 150))
        screen.blit(font_info.render(f"Depth: {depth}. Evaluation: {evaluation_method}. Simulation: {'On' if show_simulation else 'Off'}", True, BLACK), (27, SCREEN_HEIGHT - 125))
        pygame.display.flip()
        transposition_table = {} # so we can clear the transposition table for each move
        pygame.event.pump() # so the AI can think without being interrupted by other events
        
        eval_score, selected_move, *optionalisitLLMmove = ai_methods[ai_method](board, ai_depth, ai, ai, float('-inf'), float('inf'),show_simulation, actual_last_move, ai_depth)

        #print("The path to get here: ", selected_move)
       
        movie_moves = selected_move

        if selected_move:
            
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
            print(f"AI selects move: {piece} {selected_move}, eval score: {eval_score:.2f}, time: {end_time - start_time:.2f} seconds")
            pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 75, SCREEN_WIDTH - 50,25))
            screen.blit(font_info.render(f"AI: {notation}, {optionalisitLLMmove}, Ev: {eval_score:.2f}, Moves considered: {len(transposition_table)}, Time: {end_time - start_time:.0f}, Path: {len(movie_moves)} ", True, BLACK), (27, SCREEN_HEIGHT - 75))
            actual_last_move = selected_move #Track the last move added to use one call for both making and simulating.
            draw_board_wrapper(screen, board)
            draw_pieces_not_on_board(screen, board, height=SCREEN_HEIGHT)
            pygame.display.flip() 
            read_aloud("AI moves " + notation)
            read_aloud("your turn")
        
            list_of_boards = list_of_boards[:move_number+1]
            list_of_boards.append(copy.deepcopy(board))
            has_moved_history.append(copy.deepcopy(has_moved))
            player_turn = True  
            if is_in_check(board, player):
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,50))
                print(f"{piece_dict[player]} player  is in check.")
                screen.blit(font.render(f"{player} is in check.", True, BLACK), (27, SCREEN_HEIGHT - 200))
                read_aloud(f"{piece_dict[player]} is in check.")
                pygame.display.flip()
            if is_checkmate(board, player):
                print(f"Checkmate. AI {piece_dict[ai]} wins.")
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))    
                screen.blit(font.render(f"Checkmate. {piece_dict[ai]} AI wins.", True, BLACK), (27, SCREEN_HEIGHT - 50))
                read_aloud(f"Checkmate. {piece_dict[ai]} AI wins.")
                end_of_game = True
                pygame.display.flip() 
            if is_automatic_draw():
                print("Fivefold repetition. Automatic draw.")
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))    
                screen.blit(font.render("Fivefold repetition. Game is a draw.", True, BLACK), (27, SCREEN_HEIGHT - 50))
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
                    read_aloud("Threefold repetition. AI claims draw. Game over.")
                    end_of_game = True
                    pygame.display.flip()
                else:
                    print("AI chooses not to claim draw.")
                    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50, 50))    
                    screen.blit(font.render("AI chooses not to claim draw.", True, BLACK), (27, SCREEN_HEIGHT - 50))
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
            read_aloud("Hit 'r' to restart the game.")
            pygame.display.set_caption("Hit 'r' to restart the game.")
            pygame.display.flip()
            auto_save = True
            player_turn = True
pygame.quit()
sys.exit()