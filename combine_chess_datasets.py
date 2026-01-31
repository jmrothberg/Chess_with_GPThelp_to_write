#!/usr/bin/env python3
"""
Chess Dataset Combiner

Combines multiple chess .txt files into larger training sets.
Groups files in chunks of N (default 10) for better training diversity.

Usage:
    python combine_chess_datasets.py [group_size] [input_dir] [output_dir]

Example:
    python combine_chess_datasets.py 10 chess_txt_promotion_win/ combined_datasets/
"""

import os
import sys
import glob
from pathlib import Path

def combine_chess_files(input_dir: str, output_dir: str, group_size: int = 10):
    """
    Combine chess .txt files into larger groups.

    Args:
        input_dir: Directory containing chess_*.txt files
        output_dir: Directory to save combined files
        group_size: Number of files to combine in each group
    """

    # Find all chess txt files
    pattern = os.path.join(input_dir, "chess_*.txt")
    chess_files = sorted(glob.glob(pattern))

    if not chess_files:
        print(f"No chess_*.txt files found in {input_dir}")
        return

    print(f"Found {len(chess_files)} chess files:")
    for f in chess_files[:5]:  # Show first 5
        print(f"  {os.path.basename(f)}")
    if len(chess_files) > 5:
        print(f"  ... and {len(chess_files) - 5} more")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Group files into chunks
    for i in range(0, len(chess_files), group_size):
        group_files = chess_files[i:i + group_size]
        group_start = os.path.basename(group_files[0]).replace('chess_', '').replace('.txt', '')
        group_end = os.path.basename(group_files[-1]).replace('chess_', '').replace('.txt', '')

        output_file = os.path.join(output_dir, f"combined_{group_start}_{group_end}.txt")

        print(f"\nCombining group {i//group_size + 1}: {group_start} to {group_end}")
        print(f"  Files: {len(group_files)}")
        print(f"  Output: {os.path.basename(output_file)}")

        # Combine files
        total_games = 0
        total_moves = 0

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for file_path in group_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        if content.strip():  # Only write non-empty content
                            outfile.write(content)
                            # Count games (rough estimate)
                            games_in_file = len(content.split('\n\n'))
                            total_games += games_in_file
                            # Count moves (very rough estimate)
                            total_moves += len(content.split())
                except Exception as e:
                    print(f"  Warning: Failed to read {os.path.basename(file_path)}: {e}")

        print(f"  Combined: ~{total_games:,} games, ~{total_moves:,} moves")

    print(f"\n✅ Created {len(range(0, len(chess_files), group_size))} combined datasets in {output_dir}")

def main():
    # Default parameters
    group_size = 5
    input_dir = "chess_txt_promotion_win"
    output_dir = "combined_chess_datasets"

    # Parse command line arguments
    if len(sys.argv) >= 2:
        try:
            group_size = int(sys.argv[1])
        except ValueError:
            print(f"Invalid group size: {sys.argv[1]}, using default {group_size}")

    if len(sys.argv) >= 3:
        input_dir = sys.argv[2]

    if len(sys.argv) >= 4:
        output_dir = sys.argv[3]

    print("Chess Dataset Combiner")
    print("=" * 50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Group size: {group_size} files per group")
    print()

    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        print("Please run from the Chess directory or specify the correct path.")
        sys.exit(1)

    combine_chess_files(input_dir, output_dir, group_size)

if __name__ == "__main__":
    main() 
