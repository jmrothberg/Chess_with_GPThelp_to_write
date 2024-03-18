#helper for jmr chess to select games
import os
import sys
import glob
import json 
import pygame
import string
import time


# Constants for the game
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
BOARD_SIZE = 8
SQUARE_SIZE = SCREEN_WIDTH // BOARD_SIZE
GRAY = (128, 128, 128)
LIGHT_GRAY = (192, 192, 192)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Initialize the screen with given dimensions
# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Select a Saved Game")


font_info = pygame.font.SysFont("Arial", 16)
font_big = pygame.font.Font("ARIALUNI.TTF", 58)
font = pygame.font.Font("ARIALUNI.TTF", 12)

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

pieces_uni = {
    "BR1": u'\u265C',"BR2": u'\u265C', "BN": u'\u265E', "BB": u'\u265D', "BQ": u'\u265B', "BK": u'\u265A', "BP": u'\u265F',
    "WP": u'\u2659', "WR1": u'\u2656',"WR2": u'\u2656', "WN": u'\u2658', "WB": u'\u2657', "WQ": u'\u2655', "WK": u'\u2654'
}

piece_dict = {"BR1": "Black Rook","BR2": "Black Rook", "BN": "Black Knight", "BB": "Black Bishop", "BQ": "Black Queen", "BK": "Black King", "BP": "Black Pawn",
              "WR1": "White Rook", "WR2": "White Rook","WN": "White Knight", "WB": "White Bishop", "WQ": "White Queen", "WK": "White King", "WP": "White Pawn", "W": "White", "B": "Black"}

def handle_input(position):
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                return "exit"
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if SCREEN_HEIGHT - 200 <= y < SCREEN_HEIGHT:
                position[0] = (x - 24) // (icon_size + 4)
                position[1] = (y - (SCREEN_HEIGHT - 196)) // (icon_size +4)
                return position
    return None  # Return None if no selection was made

def create_icon(board, icon_size, font_size):
    font = pygame.font.Font("ARIALUNI.TTF", font_size)
    icon = pygame.Surface((icon_size, icon_size))
    square_size = icon_size // 8
    for row in range(8):
        for col in range(8):
            color = BLUE if (row + col) % 2 == 0 else GRAY
            pygame.draw.rect(icon, color, (col * square_size, row * square_size, square_size, square_size))
            piece = board[row][col]
            if piece:
                text = font.render(pieces_uni[piece], True, (WHITE) if piece[0] == "W" else (BLACK))
                icon.blit(text, (col * square_size + 0, row * square_size - 2))
    return icon


boards =[]
infos = []
def display_saved_games():
    global boards, infos, icons_per_row, icon_size
    files = glob.glob("Saved Games/board_*.json")
    files.sort(key=os.path.getmtime)
    num_files = len(files)
    if len(files) >48:
        files = files[-48:]
    if num_files <= 3:
        icon_size = 176
        font_size = 22
        icons_per_row = 3
        num_rows = 1
    elif num_files <= 12:
        icon_size = 88
        font_size = 11
        icons_per_row = 6
        num_rows = 2
    else:
        icon_size = 40
        font_size = 6
        icons_per_row = 12
        num_rows = 4
    for i, file in enumerate(files):
        with open(file, "r") as f:
            info = json.load(f)
            temp_board = info["board"]
            boards.append(temp_board)
            infos.append(info)
        icon = create_icon(boards[i], icon_size, font_size)
        x = (i % icons_per_row) * (icon_size+4) + 24
        y = (i // icons_per_row) * (icon_size+4) + 4 + 600
        screen.blit(icon, (x, y))
        
    pygame.display.flip()


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


# Function to draw the chess pieces on the board
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


def draw_pieces_not_on_board(screen, board):
    # Create a full set of chess pieces for both black and white
    full_set = {"B": ["BR1", "BN", "BB", "BQ", "BK", "BR2", "BN", "BB"]  + ["BP"] * 8,
                "W": ["WR1", "WN", "WB", "WQ", "WK", "WR2", "WN", "WB"]  + ["WP"] * 8}

    font = pygame.font.Font("ARIALUNI.TTF", 20)
    size_div = 1
    # Iterate over the board to find the pieces that are currently on the board
    for row in board:
        for piece in row:
            if piece:
                # Check if the piece is in the full set before removing it
                if piece in full_set[piece[0]]:
                    full_set[piece[0]].remove(piece)

    pygame.draw.rect(screen, LIGHT_GRAY, (0, SCREEN_HEIGHT - 200, 24,200))
    pygame.draw.rect(screen, LIGHT_GRAY, (SCREEN_WIDTH - 24, SCREEN_HEIGHT - 200, 24,200))
    # Draw the pieces not on the board
    for color, pieces in full_set.items():
        if len(pieces) > 8:
            size_div = 2
            font = pygame.font.Font("ARIALUNI.TTF", 20/size_div)
        for i, piece in enumerate(pieces):
            if color == "B":
                x = 0 + 4
                y = 604 + i * 22/size_div
            else:
                x = 600 - 24/size_div
                y = 604 + i * 22/size_div

            if piece[0] == "W":
                text = font.render(pieces_uni[piece], True, WHITE)
            else:
                text = font.render(pieces_uni[piece], True, BLACK)
            screen.blit(text, (x, y))
        

def loop_to_select_new_game(screen, board):
    pygame.display.set_caption("Use Mouse to Select, Hit return to exit")
    position = [0, 0]
    selected_info = None
    selected_game = None
    choosing_game = True
    pygame.draw.rect(screen, LIGHT_GRAY, (0, SCREEN_HEIGHT - 200, SCREEN_WIDTH,200))
    
    
    display_saved_games()
    len_boards = len(boards)
    while choosing_game:
        pygame.time.wait(100)
        new_position = handle_input(position)
        #print(new_position)
        if new_position is not None and new_position != "exit":
            
            number = position[0] + position[1] * icons_per_row
            if number < len_boards:
                pygame.draw.rect(screen, LIGHT_GRAY, (0, SCREEN_HEIGHT - 200, 24,200))
                pygame.draw.rect(screen, LIGHT_GRAY, (SCREEN_WIDTH - 24, SCREEN_HEIGHT - 200, 24,200))
                
                position = new_position
                selected_game = boards[number]
                selected_info = infos[number]
                draw_board(screen)
                draw_pieces(screen, selected_game)
                draw_pieces_not_on_board(screen, selected_game)
                #display_saved_games()
                pygame.display.flip()
        if new_position == "exit":
            pygame.draw.rect(screen, WHITE, (0, SCREEN_HEIGHT - 200, SCREEN_WIDTH,200))
            pygame.display.flip()
            return selected_info
    

if __name__ == "__main__":
    loop_to_select_new_game(screen, board)
    pygame.quit()
    sys.exit(0)
