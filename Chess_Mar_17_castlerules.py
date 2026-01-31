#JMR Simple Chess Game March 4 2024
# March 7 added en passant, castling, and pawn promotion
# March 8 refactored so make_move and simulate move ONE function and allow switching sides
# March 9 added additional evaluation functions, simpler to understand search function
# March 10 added selection of AI search function, added transposition table, sorting of moves
# March 11 working AI search, including alpha beta pruning, transposition table, and sorting of moves, and triming deeper moves
# March 12 cleaned up sorting of moves, added depth formula, cleaned up check for stalemate & checkmate
# March 13 Testing, corrected transposition table
# march 14 added tracking of path to best move, so and ability to play AI best path.
# March 15 added ability to save games, display saved games, and select a game to play
# March 16 added display of captured pieces 
# March 17 can replay the best move logig
import sys
import pygame
import copy
import time
import os
import string
import json

from game_selector import loop_to_select_new_game, draw_pieces_not_on_board

# Initialize Pygame
pygame.init()

# Constants for the game
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
BOARD_SIZE = 8
SQUARE_SIZE = SCREEN_WIDTH // BOARD_SIZE
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Initialize the screen with given dimensions
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
pygame.display.set_caption("JMR's Game of Chess Game: Press H for Help")
font = pygame.font.SysFont("Arial", 28)
font_moves = pygame.font.SysFont("Arial", 22)
font_info = pygame.font.SysFont("Arial", 16)
font_big = pygame.font.Font("ARIALUNI.TTF", 58)

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

# Helper function to convert mouse position to board coordinates
def screen_to_board_pos(mouse_pos):
    x, y = mouse_pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col

def save_game(board, move_number, player, ai, depth, evaluation_method, ai_method, depth_equation, show_simulation, list_of_boards):
    os.makedirs("Saved Games", exist_ok=True)
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    print (f"Saved game as board_{datetime}.json")
    with open(f"Saved Games/board_{datetime}.json", "w") as f:
            json.dump({"board": board, "move_number": move_number, "player": player, "ai": ai, "depth": depth, "evaluation_method": evaluation_method, "ai_method": ai_method, "depth_equation": depth_equation, "show_simulation": show_simulation, "list_of_boards": list_of_boards}, f)

# Does all moves BUT not promotions! so if only use one function would need to do promotions
def get_moves_for_piece(board, start_row, start_col, last_move = None, check_castling = True):
    
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

def is_square_under_attack(board, row, col, attacker_color):
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = board[r][c]
            if piece.startswith(attacker_color):
                moves = get_moves_for_piece(board, r, c, last_move=None, check_castling=False)
                if any(end == (row, col) for _, end in moves):
                    return True
    return False

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

def is_in_check(board, color):
    king_position = next(((r, c) for r in range(BOARD_SIZE)
                          for c in range(BOARD_SIZE)
                          if board[r][c] == f"{color}K"), None)
    return is_square_under_attack(board, king_position[0], king_position[1], "W" if color == "B" else "B")

# Moved board back but without using simulation since we know all the moves are legal. just need to check if the king is in check
def is_move_legal(board, move, color):
    legal_board = copy.deepcopy(board) # since not using siulation that copies the board
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




# Function that generates new board states given a move WITH pawn promotion
# It is also used in checking for moving into check in if move is legal in which case it just reports back all the potential captures
def simulate_move(board, move, real_board=False):
    
    start, end = move
    new_board = copy.deepcopy(board)

    piece = new_board[start[0]][start[1]]
    captured_piece = new_board[end[0]][end[1]] #en passant captures are handled below otherwsise it does pawn promostions as captured pieces

    if real_board or show_simulation:
        notation = convert_to_chess_notation(board, move)

    if real_board:
        pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,50))
        pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))
        
    # Immediately handle pawn promotion if the pawn reaches the last row
    if piece[1] == 'P':
        if (piece[0] == 'W' and end[0] == 0) or (piece[0] == 'B' and end[0] == BOARD_SIZE - 1):
            new_board[end[0]][end[1]] = piece[0] + 'Q'  # Promote to queen

            if show_simulation:
                print(f"Simulated {notation}. {piece_dict[piece]} promoted to queen")
            if real_board:
                screen.blit(font.render(f"{notation}. {piece_dict[piece]} promoted to queen", True, BLACK), (27, 615)) # put on top since don't want to cover captured piece below if also captured
                read_aloud(f"{notation}. {piece_dict[piece]} promoted to queen")
                piece = piece[0] + 'Q'  # Update the piece to the promoted piece

    # Selects en passant move need to clear the correct pawn square
    # Check if legal in the legal moves function based on the last move
    if piece[1] == 'P' and abs(start[1] - end[1]) == 1 and new_board[end[0]][end[1]] == "":
        #check color of the capturing pawn so you can clear the correct square
        if piece[0] == "W":
            if show_simulation:
                print(f"Simulated en passant capture. {notation}.")
            if real_board:
                screen.blit(font.render(f"en passant capture. {notation},  ", True, BLACK), (27, 765))
                read_aloud(f"en passant capture. {notation}.")
            new_board[end[0]+1][end[1]] = "" # clear the square behind the pawn
        else:
            if show_simulation:
                print(f"Simulated en passant capture. {notation}.")
            if real_board:
                screen.blit(font.render(f"en passant capture. {notation}.", True, BLACK), (27, 765))
                read_aloud(f"en passant capture. {notation}.")
            new_board[end[0]-1][end[1]] = "" # clear the square behind the pawn but don't move the pawn until later
            
    # Handle castling
    if piece[1] == 'K' and abs(start[1] - end[1]) == 2:
        # Determine which color is castling
        color = piece[0]
        if end[1] == start[1]+2:  # King-side castling
            new_board[start[0]][start[1]+1] = color + 'R2'
            new_board[start[0]][start[1]+3] = ''
            if show_simulation:
                print(f"Simulated {piece_dict[color]} castling.")
            if real_board:
                screen.blit(font_info.render(f"{piece_dict[color]} king-side castling.", True, BLACK), (27, 775))
                read_aloud(f"{piece_dict[color]} king-side castling.")
        elif end[1] == start[1]-2:  # Queen-side castling
            new_board[start[0]][start[1]-1] = color + 'R1'
            new_board[start[0]][start[1]-4] = '' 
            if show_simulation:
                print(f"Simulated {piece_dict[color]} castling.")   
            if real_board:
                screen.blit(font_info.render(f"{piece_dict[color]} queen-side castling.", True, BLACK), (27, 775))  
                read_aloud(f"{piece_dict[color]} queen-side castling.")
    # Capture logic 
    if captured_piece != "": # en passant captures are handled above
        if show_simulation:
            if captured_piece[1] == 'K':   # only when no checking legal. will see if works.
                print(f"Simulated Checkmate of {captured_piece}")
            else:
                print(f"Simulated {piece} captures {captured_piece} at {end}")
        if real_board:
            screen.blit(font.render(f"{piece_dict[piece]} captures {piece_dict [captured_piece]}", True, BLACK), (27, 765))
            read_aloud(f"{piece_dict[piece]} captures {piece_dict [captured_piece]}")
   
    # Perform the move
    new_board[end[0]][end[1]] = piece
    new_board[start[0]][start[1]] = ""       

    if show_simulation:
        draw_board(screen)
        draw_pieces(screen, new_board)
        pygame.display.flip()
    return new_board


def select_best_ai_move_sort(old_board, depth, color, AI_color, alpha=float('-inf'), beta=float('inf'), display_simulation=False, last_move=None, initial_depth=None):
    global transposition_table
    board_key = tuple(map(tuple, old_board))
    if board_key in transposition_table:
        eval_board,eval_depth, best_move_new = transposition_table[board_key]
        new_eval_board = evaluate_board(old_board) * (discount**((initial_depth +1) - depth))
        if eval_board != new_eval_board and display_simulation:
            print (f"Global transposition, Older eval {eval_depth}, {eval_board:.2f}, new eval {new_eval_board:.2f}, {depth}, {color}, {last_move}")
        return new_eval_board, best_move_new

    board = copy.deepcopy(old_board)
    # Initialize the dictionary
    board_table = {}

    # Initialize the list of scored moves
    scored_moves = []

    #Checks if MOVING gets you into check and is none if it doesn't get you out of check
    legal_moves = get_all_legal_moves(board, color, last_move=last_move,check_legality=True) #if you limit to legal you miss taking the king in initial score!
    
    #Are you in check?
    check = is_in_check(board, color) # get all opponents moves and see if any of them end in your king's square

    evaluation = evaluate_board(board)

    #Avoid stalemate, is in check gets oppenents legal moves
    if not check and not legal_moves:
        evaluation = evaluation * (discount**((initial_depth +1 )- depth))
        print(f"Potential Stalemate, no legal moves for {color}. Evaluation: {-evaluation:.2f}")
        print (f"Depth: {depth}, Last move: {last_move}")
        return -evaluation, [f"End stalemate {color} {depth}"]

    # Checkmate if in check and no legal moves  #since you never really take the king when legal moves are none!
    if check and not legal_moves:
        if color == "W":
            evaluation = (evaluation  - 100) * (discount**((initial_depth+1) - depth))
        else:
            evaluation = (evaluation + 100) * (discount**((initial_depth +1 ) - depth))
        print(f"Potential Checkmate of {color}. Evaluation: {evaluation:.2f}")
        print (f"Depth: {depth}, Last move: {last_move} of {'w' if color == 'B' else 'B'}")
        return evaluation, [f"End Checkmate {color} {depth}"]
    
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

            eval_board, opponent_best_move = select_best_ai_move_sort(new_board, depth-1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            
            if (color == AI_color and eval_board < best_eval) or (color != AI_color and eval_board > best_eval):
                best_eval = eval_board
                best_move_new.append(move)
                if opponent_best_move != []:
                    best_move_new.extend(opponent_best_move) 
                #best_move = move

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
            
            eval_board, opponent_best_move = select_best_ai_move_sort(new_board, depth-1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            
            if (color == AI_color and eval_board > best_eval) or (color != AI_color and eval_board < best_eval):
                best_eval = eval_board
                best_move_new.append(move)
                if opponent_best_move != []:
                    best_move_new.extend(opponent_best_move)
                #best_move = move

            if color == AI_color:
                beta = max(beta, best_eval) 
            else:
                alpha = min(alpha, best_eval) 
            
            transposition_table[tuple(map(tuple, new_board))] = best_eval, depth, best_move_new

            if beta <= alpha:
                #print(f"Break beta <= alpha, Depth: {depth}, Color: {color}, after alpha return Best move: {best_move}, value: {best_eval}")
                break

    return best_eval, best_move_new

def select_best_ai_move_sort_hard_code(old_board, depth, color, AI_color, alpha=float('-inf'), beta=float('inf'), display_simulation=False, last_move=None, initial_depth=None):
    global transposition_table

    board_key = tuple(map(tuple, old_board))
    if board_key in transposition_table:
        eval_board,eval_depth, best_move_new = transposition_table[board_key]
        new_eval_board = evaluate_board(old_board) * (discount**((initial_depth +1) - depth))
        if eval_board != new_eval_board and display_simulation:
            print (f"Global transposition, Older eval {eval_depth}, {eval_board:.2f}, new eval {new_eval_board:.2f}, {depth}, {color}, {last_move}")
        return new_eval_board, best_move_new

    board = copy.deepcopy(old_board)
    # Initialize the dictionary
    board_table = {}

    # Initialize the list of scored moves
    scored_moves = []

    #Checks if MOVING gets you into check and is none if it doesn't get you out of check
    legal_moves = get_all_legal_moves(board, color, last_move=last_move,check_legality=True) #if you limit to legal you miss taking the king in initial score!
    
    #Are you in check?
    check = is_in_check(board, color) # get all opponents moves and see if any of them end in your king's square

    evaluation = evaluate_board(board)

    #Avoid stalemate, is in check gets oppenents legal moves
    if not check and not legal_moves:
        evaluation = evaluation * (discount**((initial_depth +1 )- depth))
        print(f"Potential Stalemate, no legal moves for {color}. Evaluation: {-evaluation:.2f}")
        print (f"Depth: {depth}, Last move: {last_move}")
        return -evaluation, []

    # Checkmate if in check and no legal moves  #since you never really take the king when legal moves are none!
    if check and not legal_moves:
        if color == "W":
            evaluation = (evaluation  - 100) * (discount**((initial_depth+1) - depth))
        else:
            evaluation = (evaluation + 100) * (discount**((initial_depth +1 ) - depth))
        print(f"Potential Checkmate of {color}. Evaluation: {evaluation:.2f}")
        print (f"Depth: {depth}, Last move: {last_move} of {'w' if color == 'B' else 'B'}")
        return evaluation, []
    
    if depth == 0:
        evaluation = evaluation * (discount**((initial_depth +1) - depth))
        return evaluation, []

    for move in legal_moves:
        new_board = simulate_move(board, move, real_board=False)
        board_table[(tuple(map(tuple, board)), move)] = new_board
        score = evaluate_board(new_board) 
        scored_moves.append((move, score))

    if color == 'W':
        scored_moves.sort(key=lambda x: x[1], reverse=True)  # White maximizes
    else:
        scored_moves.sort(key=lambda x: x[1])  # Black minimizes

    if color != AI_color and initial_depth - depth > 2:
        scored_moves = scored_moves[:int(len(scored_moves)/2)]

    if AI_color == 'B':
        best_eval = float('inf') if color == AI_color else float('-inf')
        best_move_new = []
        for move, _ in scored_moves:
           
            move_key = (tuple(map(tuple, board)), move)
            if move_key in board_table:
                new_board = board_table[move_key]

            
            eval_board, opponent_best_move = select_best_ai_move_sort(new_board, depth-1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            
            if (color == AI_color and eval_board < best_eval) or (color != AI_color and eval_board > best_eval):
                best_eval = eval_board
                best_move_new.append(move)
                if opponent_best_move != []:
                    best_move_new.extend(opponent_best_move)    
                #best_move = move

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

            eval_board, opponent_best_move = select_best_ai_move_sort(new_board, depth-1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            
            if (color == AI_color and eval_board > best_eval) or (color != AI_color and eval_board < best_eval):
                best_eval = eval_board
                best_move_new.append(move)
                if opponent_best_move != []:
                    best_move_new.extend(opponent_best_move)
                #best_move = move

            if color == AI_color:
                beta = max(beta, best_eval) 
            else:
                alpha = min(alpha, best_eval) 
            
            transposition_table[tuple(map(tuple, new_board))] = best_eval, depth, best_move_new

            if beta <= alpha:
                #print(f"Break beta <= alpha, Depth: {depth}, Color: {color}, after alpha return Best move: {best_move}, value: {best_eval}")
                break

    return best_eval, best_move_new


def select_best_ai_move_min_max(old_board, depth, color, AI_color, alpha=float('-inf'), beta=float('inf'), display_simulation=False, last_move=None, initial_depth=None):
    
    board = copy.deepcopy(old_board)
    
    #Checks if MOVING gets you into check and is none if it doesn't get you out of check
    legal_moves = get_all_legal_moves(board, color, last_move=last_move,check_legality=True) #if you limit to legal you miss taking the king in initial score!
    
    #Are you in check?
    check = is_in_check(board, color) # get all opponents moves and see if any of them end in your king's square

    evaluation = evaluate_board(board)

    #Avoid stalemate, is in check gets oppenents legal moves
    if not check and not legal_moves:
        evaluation = evaluation * (discount**((initial_depth +1 )- depth))
        print(f"Potential Stalemate, no legal moves for {color}. Evaluation: {-evaluation:.2f}")
        print (f"Depth: {depth}, Last move: {last_move}")
        return -evaluation, [f"End stalemate {color} {depth}"]

    # Checkmate if in check and no legal moves  #since you never really take the king when legal moves are none!
    if check and not legal_moves:
        if color == "W":
            evaluation = (evaluation  - 100) * (discount**((initial_depth+1) - depth))
        else:
            evaluation = (evaluation + 100) * (discount**((initial_depth +1 ) - depth))
        print(f"Potential Checkmate of {color}. Evaluation: {evaluation:.2f}")
        print (f"Depth: {depth}, Last move: {last_move} of {'w' if color == 'B' else 'B'}")
        return evaluation, [f"End Checkmate {color} {depth}"]
    
    if depth == 0:
        evaluation = evaluation * (discount**((initial_depth +1) - depth))
        return evaluation, [f"End depth {color} {depth}"]

    if AI_color == 'B':
        best_move_new = []
        best_eval = float('inf') if color == AI_color else float('-inf')
        
        for move in legal_moves:
            new_board = simulate_move(board, move)
           
            eval_board, opponent_best_move = select_best_ai_move_min_max(new_board, depth - 1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            
            if (color == AI_color and eval_board < best_eval) or (color != AI_color and eval_board > best_eval):
                best_eval = eval_board
                best_move_new.append(move)
                if opponent_best_move != []:
                    best_move_new.extend(opponent_best_move)    
                #best_move = move

            if color == AI_color:
                beta = min(beta, best_eval)
            else:
                alpha = max(alpha, best_eval)

            if beta <= alpha:
                #print(f"Break beta <= alpha, Depth: {depth}, Color: {color}, after alpha return Best move: {best_move}, value: {best_eval}")
                break

    if AI_color == 'W':
        best_move_new =[]
        best_eval = float('-inf') if color == AI_color else float('inf')
        for move in legal_moves:
            new_board = simulate_move(board, move)
            eval_board, opponent_best_move = select_best_ai_move_min_max(new_board, depth - 1, 'B' if color == 'W' else 'W', AI_color, alpha, beta, display_simulation, move, initial_depth)
            
            if (color == AI_color and eval_board > best_eval) or (color != AI_color and eval_board < best_eval):
                best_eval = eval_board
                if opponent_best_move != []:
                    best_move_new.extend(opponent_best_move)    
                best_move_new.append(move)
                #best_move = move

            if color == AI_color:
                beta = max(beta, best_eval)
            else:
                alpha = min(alpha, best_eval)
            if beta <= alpha:
                #print(f"Break beta <= alpha, Depth: {depth}, Color: {color}, after alpha return Best move: {best_move}, value: {best_eval}")
                break
    return best_eval, best_move_new

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
        'WP': 1, 'WN': 3, 'WB': 3, 'WR1': 5, 'WR2': 5, 'WQ': 9, 'WK': 100,
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

def evaluate_board_center(board): #updated to for piece values also negative

    piece_values = {
        'WP': 1, 'WN': 3, 'WB': 3, 'WR1': 5, 'WR2': 5, 'WQ': 9, 'WK': 100,
        'BP': -1, 'BN': -3, 'BB': -3, 'BR1': -5, 'BR2': -5, 'BQ': -9, 'BK': -100
    }

    evaluation = 0
    for i, row in enumerate(board):
        for j, piece in enumerate(row):
            if piece:
                # Use the piece values lookup instead of repeated if-else
                value = piece_values[piece]
                # No conditional, rely on the sign being flipped for black pieces
                evaluation += value
                # Add positional adjustment using the sign of the value
                pos_val_table = positional_values.get(piece[1], [])
                if pos_val_table:
                    pos_value = pos_val_table[7-i][j] if 'W' in piece else pos_val_table[i][j]
                    evaluation += pos_value if value > 0 else -pos_value
                # Control of the center (simplified with tuple unpacking)
                if (i, j) in [(3, 3), (3, 4), (4, 3), (4, 4)]:
                    evaluation += 0.5 if value > 0 else -0.5
                # Moving pieces from the back rank
                if i in (0, 7) and 'K' not in piece and 'P' not in piece:
                    evaluation += -0.5 if value > 0 else 0.5
    return evaluation


# Reused function to apply positional value
def apply_piece_value(piece, i, j):
    piece_values = {
        'WP': 1, 'WN': 3, 'WB': 3, 'WR1': 5, 'WR2': 5, 'WQ': 9, 'WK': 100,
        'BP': -1, 'BN': -3, 'BB': -3, 'BR1': -5, 'BR2': -5, 'BQ': -9, 'BK': -100
    }

    pos_val = positional_values[piece[1]][i][j] * (-1 if piece.startswith('B') else 1)
    return piece_values[piece] + pos_val


def evaluate_board_structure_optimized(board):
    evaluation = 0
    white_development_penalty = 0
    black_development_penalty = 0

    # Pre-calculated center control adjustments
    center_squares_values = {
    (3, 3): 0.5, (3, 4): 0.5, (4, 3): 0.5, (4, 4): 0.5
    }
    
    # Calculate pawn structure inside the main loop
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece:
                # Adjust evaluation based on piece value and positional value
                evaluation += apply_piece_value(piece, i, j)
                
                # Adjust for center control
                if (i, j) in center_squares_values:
                    evaluation += center_squares_values[(i, j)] if piece.startswith('W') else -center_squares_values[(i, j)]

                # Adjust evaluation for undeveloped pieces
                if piece in ('WR', 'WN', 'WB') and i == 7:
                    white_development_penalty += 0.5
                elif piece in ('BR', 'BN', 'BB') and i == 0:
                    black_development_penalty += 0.5

                # Adjust evaluation for pawn structure
                if piece[1] == 'P':
                    isolated = (j == 0 or (board[i][j-1] and board[i][j-1][1] != 'P')) and (j == 7 or (board[i][j+1] and board[i][j+1][1] != 'P'))
                    doubled = (i < 7 and board[i+1][j] == piece) or (i > 0 and board[i-1][j] == piece)
                    blocked = (i < 7 and board[i+1][j] and board[i+1][j] != '' and board[i+1][j][1] != 'P') or (i > 0 and board[i-1][j] and board[i-1][j] != '' and board[i-1][j][1] != 'P')
                    if isolated or doubled or blocked:
                        evaluation -= 0.5 if piece.startswith('W') else 0.5

                # Add bonus for king safety (castling, pawn shield)
                if piece[1] == 'K':
                    if j == 2 or j == 6:
                        evaluation += 1 if piece.startswith('W') else -1  # castled king
                    if i in range(1, 7) and j in range(1, 7):  # if king is not on the edge of the board
                        for row_offset in [-1, 0, 1]:
                            for col_offset in [-1, 0, 1]:
                                if board[i + row_offset][j + col_offset][1] == 'P':
                                    evaluation += 0.1 if piece.startswith('W') else -0.1  # pawn shield
    
    # Apply development penalties at the end to avoid redundant checks
    evaluation -= white_development_penalty - black_development_penalty
    return evaluation


# Function to draw the chess board
def draw_board(screen):
    font = pygame.font.SysFont("Arial", 12)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            color = BLUE if (row + col) % 2 == 0 else GRAY
            square = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, square)
            # Display the square's coordinates in the lower right corner
            start_square = string.ascii_lowercase[col] + str(8 - row)
            coords_text = font.render(start_square, True, BLACK)
            #coords_text = font.render(f"{row},{col}", True, BLACK)
            screen.blit(coords_text, (square.right - coords_text.get_width(),
                                      square.bottom - coords_text.get_height()))


def draw_pieces(screen, board):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece:
                if piece[0] == "W":
                    text = font_big.render(pieces_uni[piece], True, WHITE)
                else:
                    text = font_big.render(pieces_uni[piece], True, BLACK)
                screen.blit(text, (col * SQUARE_SIZE + 8, row * SQUARE_SIZE - 8))
            

def play_moves(board, moves, depth):
    global show_simulation
    show_simulation = True
    print("Depth, Playing moves:",depth, moves)
    draw_board(screen)
    draw_pieces(screen, board)
    pygame.display.flip()
    time_delay = int(10000/len(moves))
    for move in moves:
        print(f"Move: {move}")
        if "End" in move:
            break
        else:
            start_pos, end_pos = move   
            piece = board[start_pos[0]][start_pos[1]]
            piece_end = board[end_pos[0]][end_pos[1]]
            print(f"Piece start: {piece}, Piece end: {piece_end}")
            if piece:
                notation = convert_to_chess_notation(board, move)
                print(f"Move: {notation}")
                read_aloud(notation)
                board = simulate_move(board, move, real_board=True)
                
    show_simulation = False    
    pygame.time.wait(2*time_delay)
    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50,50))
    screen.blit(font_info.render(f"Move: {move_number}. Player: {player}.  Ai: {ai_method}. {depth_equation}", True, BLACK), (27, 650))
    return


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
    os.system("say " + text)


def initialize_game():
    # Initial game state setup
    global setauto_switch_colors_for_player, depth_formula,transposition_table, depth_equation,discount,player_turn, selected_piece, actual_last_move, \
        list_of_boards, move_number, end_of_game, running, show_simulation, board, depth, player,ai, evaluate_board, evaluation_method,select_best_ai_move, ai_method, has_moved
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

    evaluation_method = "Positions_optimized"
    evaluate_board = evaluation_methods[evaluation_method]
    ai_method = "Sort_moves"
    select_best_ai_move = ai_methods[ai_method]
    depth_equation = "All .95"
    depth_formula = depth_equations[depth_equation][0]
    discount = depth_equations[depth_equation][1]

    has_moved = {"WK": False, "WR1": False, "WR2": False, "BK": False, "BR1": False, "BR2": False}

    draw_board(screen)
    draw_pieces(screen, board)  
    draw_pieces_not_on_board(screen,board)
    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,200))
    screen.blit(font_info.render(f"Move: {move_number}. Player: {player}.  Ai: {ai_method}. {depth_equation}", True, BLACK), (27, 650))
    screen.blit(font_info.render(f"Depth: {depth}. Evaluation Method {evaluation_method}. Show simulation: {show_simulation}", True, BLACK), (27, 675))
    pygame.display.flip()


def help():
    # Draw the board and pieces & Help screen
    screen.fill(WHITE)
    draw_board(screen)
    draw_pieces(screen, board)
    screen.blit(font_info.render("Use Mouse click to select move.", True, BLACK), (27, 605))
    screen.blit(font_info.render("Press 's', 'l' or 'r' save, load, restart.", True, BLACK), (27, 630))
    screen.blit(font_info.render("Press 'y' & 'n' to toggle display of AI thinking.", True, BLACK), (27, 655))
    screen.blit(font_info.render("Press up-arrow or down-arrow to set depth.", True, BLACK), (27, 680))
    screen.blit(font_info.render("Press 'a' or 'e' to select AI or evaluation formula.", True, BLACK), (27, 705))
    screen.blit(font_info.render("Press left-arrow or right-arrow to roll back/forward moves.", True, BLACK), (27, 730))
    screen.blit(font_info.render(f"Player is {player} press 'x' to switch colors for player.", True, BLACK), (27, 755))
    screen.blit(font_info.render("Press 'h' to toggle help screen.", True, BLACK), (27, 780))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,200))
                    return

 # Map user input to function
evaluation_methods = {
    'Basic': evaluate_board_basic,
    'Positions_optimized': evaluate_board_positions_optimized,
    "Center": evaluate_board_center,
    "Structure_optimized": evaluate_board_structure_optimized
    }

ai_methods = {
    'sort_hardcode 2 player/2': select_best_ai_move_sort_hard_code,
    'Sort_moves': select_best_ai_move_sort,
    'Minimax': select_best_ai_move_min_max
}

depth_equations = {
    "All": ("length", 1),
    "All .95": ("length", .95),
    "Half length 2 in .95": ("int(length/2) if depth < initial_depth - 4 else length", .95),
    "Half length 4 in .95": ("int(length/2) if depth < initial_depth - 6 else length", .95),
    "Half length 6 in .95": ("int(length/2) if depth < initial_depth - 4 else length", .95),
    "length/initial_depth +1-depth 2 in .95": ("int(length/((initial_depth +1)-depth)) if depth < initial_depth - 2 else length", .95),
    "length/initial_depth +1-depth 4 in .95": ("int(length/((initial_depth +1)-depth)) if depth < initial_depth - 4 else length", .90),
    }

# Main game loop
initialize_game()
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
                print("AI Thinking will be displayed.")
                show_simulation = True
            elif event.key == pygame.K_n:
                print("AI Thinking will not be displayed.")
                show_simulation = False
           
            if event.key == pygame.K_LEFT:
                # Go to the previous board state if it exists
                if move_number > 0:
                    move_number -= 1
                    print ("Going back to:", move_number)
                    board = copy.deepcopy(list_of_boards[move_number])
                    if move_number < len(list_of_boards) and end_of_game:
                        end_of_game = False

            elif event.key == pygame.K_RIGHT:
                # Go to the next board state if it exists
                if move_number < len(list_of_boards) - 1:
                    move_number += 1
                    print ("Going forward to: ", move_number)
                    board = copy.deepcopy(list_of_boards[move_number])

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
                index = keys.index(ai_method)
                if index < len(keys) - 1:
                    index += 1
                else:
                    index = 0
                ai_method = keys[index]
                select_best_ai_move = ai_methods[ai_method]
                print(f"AI method is now {ai_method}.")

            #change the depth equation
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

            #play the moves
            if event.key == pygame.K_p:
                if last_board and movie_moves :
                    play_moves(last_board, movie_moves, depth)

            if event.key == pygame.K_s:
                print("Saving game...", board)
                screen.blit(font_info.render("Saving game...", True, BLACK), (27, 750))
                read_aloud("Saving game")
                save_game(board, move_number, player, ai, depth, evaluation_method, ai_method, depth_equation, show_simulation, list_of_boards)
               
            if event.key == pygame.K_l:
                old_board = board
                info = loop_to_select_new_game(screen, board)
                if not info:
                    board = old_board
                    print("No game loaded.")
                    screen.blit(font_info.render("No game loaded.", True, BLACK), (27, 750))
                    read_aloud("No game loaded")
                
                # info = {"board": board, "move_number": move_number, "player": player, "ai": ai, "depth": depth, "evaluation_method": evaluation_method, "ai_method": ai_method, "depth_equation": depth_equation, "show_simulation": show_simulation, "list_of_boards": list_of_boards}, f)
                if info:
                    board = info["board"]
                    move_number = info["move_number"]
                    player = info["player"]
                    ai = info["ai"]
                    depth = info["depth"]
                    evaluation_method = info["evaluation_method"]
                    ai_method = info["ai_method"]
                    depth_equation = info["depth_equation"]
                    show_simulation = info["show_simulation"]
                    list_of_boards = info["list_of_boards"]
                    print("Loaded game...", board)
                    screen.blit(font_info.render("Loaded game...", True, BLACK), (27, 750))
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
            if event.key == pygame.K_r:
                initialize_game()
            if event.key == pygame.K_h:
                help()

            pygame.time.wait(100)
            draw_board(screen)
            draw_pieces(screen, board)  
            draw_pieces_not_on_board(screen, board)
            pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 150, SCREEN_WIDTH - 50,50))
            screen.blit(font_info.render(f"Move: {move_number}. Player: {player}.  Ai: {ai_method}. {depth_equation}", True, BLACK), (27, 650))
            screen.blit(font_info.render(f"Depth: {depth}. Evaluation Method {evaluation_method}. Show simulation: {show_simulation}", True, BLACK), (27, 675))
            pygame.display.flip()
         
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if player_turn and not end_of_game:  
                if event.pos[1] < 600:
                    pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 150, SCREEN_WIDTH - 50,50))
                    screen.blit(font_info.render(f"Move: {move_number}. Player: {player}.  Ai: {ai_method}. {depth_equation}", True, BLACK), (27, 650))
                    screen.blit(font_info.render(f"Depth: {depth}. Evaluation Method {evaluation_method}. Show simulation: {show_simulation}", True, BLACK), (27, 675))
                    pygame.display.flip()
                    pos = screen_to_board_pos(event.pos)
                    if selected_piece:
                        move = (selected_piece, pos)
                        if move in get_all_legal_moves(board, player, last_move = actual_last_move):
                            notation = convert_to_chess_notation(board, move)
                            piece = board[selected_piece[0]][selected_piece[1]]
                            print(f"Player moves: {piece} {move}")
                            pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 100, SCREEN_WIDTH - 50,50))
                            
                            screen.blit(font_info.render("Player moves: "+notation, True, BLACK), (27, 700))
                            board = simulate_move(board, move, real_board=True)
                            actual_last_move = move #Track the last move
                            #check if the piece is in the have_moved dictionary
                            if piece in has_moved:
                                has_moved[piece] = True
                            
                            draw_board(screen)
                            draw_pieces(screen, board)
                            draw_pieces_not_on_board(screen, board)
                            pygame.display.flip()
                            read_aloud("Player moves "+notation)
                            if is_checkmate(board, "B"):
                                print("Checkmate. White wins.")
                                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 100, SCREEN_WIDTH - 50,200))    
                                screen.blit(font.render("Checkmate. White wins.", True, BLACK), (27, 750))
                                screen.blit(font.render("Hit 'r' to restart the game.", True, BLACK), (27, 700))
                                end_of_game = True
                            player_turn = False
                        else:
                            print("Illegal move")
                            screen.blit(font_info.render("Illegal move", True, BLACK), (27, 750))
                            selected_piece = None
                    else:
                        if board[pos[0]][pos[1]] and board[pos[0]][pos[1]].startswith(player):
                            selected_piece = pos

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
        screen.blit(font.render(f"Self_Play.", True, BLACK), (27, 615))  
        print(f"Player is now {piece_dict [player]}.")
        read_aloud(f"Player is now {piece_dict[player]}.")
        print(f"AI is now {piece_dict[ai]}.")
        read_aloud(f"AI is now {piece_dict[ai]}.")
        
    if not player_turn and not end_of_game:
        # Adding a timer to the AI's move
        start_time = time.time()
        ai_depth = depth  # You can ask the user for this input or adjust as needed
        move_number = move_number + 1
        print(f"AI {ai} is thinking...")
        pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,100))
        screen.blit(font.render(f"AI {piece_dict[ai]} is thinking...", True, BLACK), (27, 615))
        screen.blit(font_info.render(f"Move: {move_number}. Player: {player}.  Ai: {ai_method}. {depth_equation}", True, BLACK), (27, 650))
        screen.blit(font_info.render(f"Depth: {depth}. Evaluation: {evaluation_method:}. Show simulation: {show_simulation}", True, BLACK), (27, 675))
        pygame.display.flip()
        transposition_table = {} # so we can clear the transposition table for each move
        eval_score, selected_move = select_best_ai_move(board, ai_depth, ai, ai, float('-inf'), float('inf'),show_simulation, actual_last_move, ai_depth)
        
        if show_simulation:
            print ("The path to get here: ", selected_move)
        movie_moves = selected_move
        selected_move = selected_move[0] 
        
        if selected_move:
            notation = convert_to_chess_notation(board, selected_move)
            new_position = selected_move[0]  #before the move you need to do the from :) 
            piece = board[new_position[0]][new_position[1]]
            last_board = copy.deepcopy(board) # for movie playback reducant with list of boards but good for now
            board = simulate_move(board, selected_move, real_board=True)
            end_time = time.time()
            print(f"AI selects move: {piece} {selected_move}, eval score: {eval_score:.2f}, time: {end_time - start_time:.2f} seconds")
            pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 75, SCREEN_WIDTH - 50,25))
            screen.blit(font_info.render(f"AI: {notation}, Ev: {eval_score:.2f}, Moves: {len(transposition_table)}, Time: {end_time - start_time:.0f}, Path: {len(movie_moves)} ", True, BLACK), (27, 725))
            actual_last_move = selected_move #Track the last move added to use one call for both making and simulating.
            # Drawing the board and pieces
            draw_board(screen)
            draw_pieces(screen, board)
            draw_pieces_not_on_board(screen, board)
            pygame.display.flip() 
            read_aloud("AI moves " + notation)
            os.system('say "your turn"')
            list_of_boards = list_of_boards[:move_number+1]
            list_of_boards.append(copy.deepcopy(board))
            
            if is_in_check(board, player):
                pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,50))
                print(f"{piece_dict[player]} player  is in check.")
                screen.blit(font.render(f"{player} is in check.", True, BLACK), (27, 615))
                read_aloud(f"{piece_dict[player]} is in check.")
                pygame.display.flip()
            if is_checkmate(board, player):
                print(f"Checkmate. AI {piece_dict[ai]} wins.")
                screen.blit(font.render(f"Checkmate. {piece_dict[ai]} AI wins.", True, BLACK), (27, 750))
                read_aloud(f"Checkmate. {piece_dict[ai]} AI wins.")
                end_of_game = True
                pygame.display.flip() 
            player_turn = True    
        else:
            print("No legal moves for AI")
            print("Stalemate.")
            pygame.draw.rect(screen, WHITE, (25, SCREEN_HEIGHT - 200, SCREEN_WIDTH - 50,200))
            screen.blit(font.render(f"No legal moves for {piece_dict[ai]} AI  ", True, BLACK), (27, 615))
            read_aloud(f"No legal moves for {piece_dict[ai]} AI  ")
            screen.blit(font.render("Stalemate.", True, BLACK), (27, 675))
            read_aloud("Stalemate.")
            print("Hit 'r' to restart the game.")
            screen.blit(font.render("Hit 'r' to restart the game.", True, BLACK), (27, 700))
            read_aloud("Hit 'r' to restart the game.")
            pygame.display.flip() 
            end_of_game = True
    if end_of_game:
        print("Saving game...", board)
        screen.blit(font_info.render("Saving game...", True, BLACK), (27, 750))
        read_aloud("Saving game")
        save_game(board, move_number, player, ai, depth, evaluation_method, ai_method, depth_equation, show_simulation, list_of_boards)
        pygame.display.set_caption("Hit 'r' to restart the game.")
pygame.quit()
sys.exit()