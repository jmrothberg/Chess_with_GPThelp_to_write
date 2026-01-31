# JMR parquet to chess txt
# This script converts Parquet files containing chess game data into text files.
#
# TWO MODES:
# 1. OLD MODE (GUI): Generic text extraction from any column (legacy Brain6 format)
# 2. NEW MODE (Default): UCI + promotions + win/draw/loss tokens (factorized training)
#
# Default behavior: Batch-convert chess_parquet/ -> chess_txt_promotion_win/
# preserving promotions (e7e8q) and adding result markers (<W>/<D>/<L>).
#
# https://huggingface.co/datasets/laion/strategic_game_chess

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import re

# --------------------------------------------------------------------------------------
# OLD MODE (Legacy): Generic text extraction (works with any column format)
# --------------------------------------------------------------------------------------

def convert_parquet_to_text(input_parquet, output_text, selected_column):
    # Read the Parquet file
    df = pd.read_parquet(input_parquet)
    
    # Initialize statistics
    total_games = 0
    total_moves = 0
    total_entry_length = 0  # New: track total length of processed entries
    total_items = 0        # New: track total number of items per entry
    longest_game = 0
    shortest_game = float('inf')
    
    # Print column names
    print("Available columns:", df.columns.tolist())
    
    # Open the output text file
    with open(output_text, 'w') as f:
        # Iterate through each row (game) in the DataFrame
        for index, row in df.iterrows():
            # Get the moves
            moves = row[selected_column]
            
            # Process moves: convert to string, remove move numbers and join with spaces
            if isinstance(moves, np.ndarray):
                processed_moves = ' '.join(move.split('.')[-1].strip() for move in moves if isinstance(move, str))
                items_count = len(moves)  # New: count array items
            elif isinstance(moves, str):
                processed_moves = ' '.join(move.split('.')[-1].strip() for move in moves.split())
                items_count = len(moves.split())  # New: count space-separated items
            else:
                print(f"Unexpected data type for moves in row {index}: {type(moves)}")
                continue
            
            # Update statistics
            move_count = len(processed_moves.split())
            total_games += 1
            total_moves += move_count
            total_items += items_count           # New: accumulate items count
            total_entry_length += len(processed_moves)  # New: accumulate processed length
            longest_game = max(longest_game, move_count)
            shortest_game = min(shortest_game, move_count)
            
            # Write the processed moves to the file
            f.write(processed_moves)
            f.write('\n\n')
            
            # Debug: Print the first 5 entries
            if index < 5:
                print(f"Game {index + 1}:")
                print(f"Original moves: {moves[:100]}...")  # Print first 100 characters
                print(f"Processed moves: {processed_moves[:100]}...")
                print()

    # Return enhanced statistics
    return {
        'total_games': total_games,
        'total_moves': total_moves,
        'avg_moves_per_game': total_moves / total_games if total_games > 0 else 0,
        'avg_items_per_entry': total_items / total_games if total_games > 0 else 0,  # New
        'avg_entry_length': total_entry_length / total_games if total_games > 0 else 0,  # New
        'longest_game': longest_game,
        'shortest_game': shortest_game
    }


# --------------------------------------------------------------------------------------
# NEW MODE (Dec 2025): UCI format with promotions + win/draw/loss tokens
# For factorized training (FROM/TO/PROMO heads + value learning)
# --------------------------------------------------------------------------------------

# Keep only UCI-like tokens (e2e4 or e7e8q).
_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)


def _result_to_token(result_str: str) -> str:
    """
    Map game results to a compact token for a value head:
      1-0       -> <W> (white win)
      1/2-1/2   -> <D> (draw)
      0-1       -> <L> (black win)
      otherwise -> <U> (unknown)
    """
    if result_str == "1-0":
        return "<W>"
    if result_str == "1/2-1/2":
        return "<D>"
    if result_str == "0-1":
        return "<L>"
    return "<U>"


def convert_parquet_to_uci_with_result(input_parquet: str, output_text: str,
                                       moves_col: str = "Moves", result_col: str = "Result"):
    """
    Export a parquet into a consistent chess text format:
    - Each game is separated by a blank line
    - Each game begins with a result token (<W>/<D>/<L>/<U>)
    - Then space-separated UCI moves, preserving promotions (5th char when present)
    """
    # NOTE: We explicitly select the two columns we need for speed/memory.
    df = pd.read_parquet(input_parquet, columns=[moves_col, result_col])

    total_games = 0
    total_moves = 0
    bad_rows = 0

    with open(output_text, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            moves = row.get(moves_col, None)
            result = str(row.get(result_col, "")).strip()

            if not isinstance(moves, (list, tuple, np.ndarray)):
                bad_rows += 1
                continue

            cleaned = []
            for m in moves:
                if not isinstance(m, str):
                    continue
                m = m.strip()
                if _UCI_RE.match(m):
                    cleaned.append(m.upper())

            if not cleaned:
                bad_rows += 1
                continue

            total_games += 1
            total_moves += len(cleaned)

            f.write(_result_to_token(result))
            f.write(" ")
            f.write(" ".join(cleaned))
            f.write("\n\n")

    print(f"\nWrote: {output_text}")
    print(f"Games: {total_games:,} | Moves: {total_moves:,} | Bad/empty rows skipped: {bad_rows:,}")


def convert_parquet_folder_to_chess_txt_promotion_win(input_folder: str, output_folder: str):
    """
    Batch-convert all *.parquet files in a folder into consistent UCI+Result txt files.
    Output naming: chess_<original parquet name>.txt
    """
    os.makedirs(output_folder, exist_ok=True)
    parquet_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".parquet")]
    parquet_files.sort()

    if not parquet_files:
        print(f"No parquet files found in: {input_folder}")
        return

    for fname in parquet_files:
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, f"chess_{fname.replace('.parquet', '.txt')}")
        convert_parquet_to_uci_with_result(in_path, out_path)

if __name__ == "__main__":
    # DEFAULT BEHAVIOR: New factorized training format
    # Converts chess_parquet/ -> chess_txt_promotion_win/ with:
    # - UCI moves (preserving promotions like e7e8q)
    # - Result tokens (<W>/<D>/<L>) for value learning
    input_folder = "/home/jonathan/Chess/chess_parquet"
    output_folder = "/home/jonathan/Chess/chess_txt_promotion_win"

    if os.path.isdir(input_folder):
        print(f"🆕 NEW MODE: Converting {input_folder} -> {output_folder}")
        print("   → Preserving UCI promotions (e7e8q)")
        print("   → Adding win/draw/loss tokens (<W>/<D>/<L>)")
        print("   → For factorized training (FROM/TO/PROMO + value)")
        convert_parquet_folder_to_chess_txt_promotion_win(input_folder, output_folder)
    else:
        print(f"⚠️  Input folder not found: {input_folder}")
        print("🖥️  Falling back to OLD GUI mode for backwards compatibility")
        root = tk.Tk()
        root.withdraw()

        input_files = filedialog.askopenfilenames(title="Select Parquet files", filetypes=[("Parquet files", "*.parquet")])
        output_directory = filedialog.askdirectory(title="Select output directory")

        if input_files and output_directory:
            sample_df = pd.read_parquet(input_files[0])
            columns = sample_df.columns.tolist()

            selection_window = tk.Toplevel()
            selection_window.title("Select Column")

            tk.Label(selection_window, text="Select column to process:").pack()
            column_var = tk.StringVar(value=columns[0])
            column_menu = tk.OptionMenu(selection_window, column_var, *columns)
            column_menu.pack()

            tk.Label(selection_window, text="Enter output file name (without .txt):").pack()
            name_entry = tk.Entry(selection_window)
            name_entry.pack()
            name_entry.insert(0, "chess_moves")

            def process_files():
                selected_column = column_var.get()
                output_base_name = name_entry.get()
                selection_window.destroy()

                print(f"\nProcessing files using column: {selected_column}")
                for input_file in input_files:
                    output_file = os.path.join(output_directory, f"{output_base_name}_{os.path.basename(input_file).replace('.parquet', '.txt')}")
                    stats = convert_parquet_to_text(input_file, output_file, selected_column)

                    # Enhanced statistics printing
                    print(f"\nStatistics for {os.path.basename(input_file)}:")
                    print(f"Total games processed: {stats['total_games']:,}")
                    print(f"Total moves/items: {stats['total_moves']:,}")
                    print(f"Average moves/items per game: {stats['avg_moves_per_game']:.2f}")
                    print(f"Average raw items per entry: {stats['avg_items_per_entry']:.2f}")
                    print(f"Average processed entry length: {stats['avg_entry_length']:.2f} characters")
                    print(f"Longest game: {stats['longest_game']} moves")
                    print(f"Shortest game: {stats['shortest_game']} moves")
                    print(f"Output saved to: {output_file}")

            tk.Button(selection_window, text="Process Files", command=process_files).pack()
            selection_window.mainloop()
        else:
            print("Required file selection not completed. Conversion cancelled.")