#!/usr/bin/env python3
"""
Loss plotting for ChessBrain checkpoints.
Scans a checkpoint folder and plots loss over training time.

Usage:
  python plot_loss.py <checkpoint_folder>
  python plot_loss.py  # Interactive folder selection
"""

import os
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False


def parse_checkpoint_filename(filename):
    """
    Parse checkpoint filename → epoch, batch, loss.
    Format: ..._E{epoch}B{batch}_L{loss}_{MMDD}_{HHMM}.pth
    """
    pattern = r'E(\d+)B(\d+)_L([\d.]+)_'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2)), float(match.group(3))
    return None


def scan_checkpoints(folder_path):
    """
    Scan folder for .pth checkpoints.
    Returns list of (epoch, batch, loss, mtime, filename) sorted oldest → newest.
    (Sort by file time so a new dataset that restarts at E1 stays in real order.)
    """
    checkpoints = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return []

    for filename in os.listdir(folder_path):
        if not filename.endswith('.pth'):
            continue
        data = parse_checkpoint_filename(filename)
        if not data:
            continue
        epoch, batch, loss = data
        full_path = os.path.join(folder_path, filename)
        mtime = os.path.getmtime(full_path)
        checkpoints.append((epoch, batch, loss, mtime, filename))

    checkpoints.sort(key=lambda x: (x[3], x[4]))
    return checkpoints


def build_continuous_x(checkpoints):
    """
    Epoch-like x-axis with no visual gaps.

    Uses the largest batch index seen as batches-per-epoch (e.g. ~20k), NOT a
    hardcoded 60k (that made E1 end near 1.33 then jump to E2 at 2.0).

    When training reloads data and epoch counters reset to E1, the axis keeps
    going forward from the last point instead of jumping backward.
    """
    if not checkpoints:
        return []

    bpe = float(max(cp[1] for cp in checkpoints) or 1)
    xs = []
    session_offset = 0.0
    prev_e = prev_b = None

    for epoch, batch, _loss, _mtime, _fn in checkpoints:
        local = (epoch - 1) + (batch / bpe)
        if prev_e is not None and (epoch < prev_e or (epoch == prev_e and batch < prev_b)):
            # New session (Ctrl+C → load new parquet, epoch restarts at 1)
            session_offset = xs[-1]
        x = session_offset + local
        if xs and x <= xs[-1]:
            x = xs[-1] + (1.0 / bpe)  # keep strictly increasing
        xs.append(x)
        prev_e, prev_b = epoch, batch

    return xs


def plot_loss_progression(checkpoints, folder_name):
    """Plot raw and smoothed loss vs continuous training progress."""
    if not checkpoints:
        print("No valid checkpoints found!")
        return

    losses = [cp[2] for cp in checkpoints]
    x_values = build_continuous_x(checkpoints)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x_values, losses, 'b-', marker='o', markersize=3, linewidth=1)
    plt.xlabel('Training progress (epoch units, continuous across data reloads)')
    plt.ylabel('Loss')
    plt.title(f'Loss Progression - {folder_name}')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    window_size = min(50, len(losses))
    smoothed_losses = []
    smoothed_x = []
    for i in range(len(losses)):
        if i >= window_size - 1:
            window_avg = sum(losses[i - window_size + 1:i + 1]) / window_size
            smoothed_losses.append(window_avg)
            smoothed_x.append(x_values[i])

    plt.plot(smoothed_x, smoothed_losses, 'r-', linewidth=2)
    plt.xlabel('Training progress (epoch units, continuous across data reloads)')
    plt.ylabel(f'Smoothed Loss ({window_size}-point average)')
    plt.title(f'Smoothed Loss Progression - {folder_name}')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def select_folder_interactive():
    if not HAS_TKINTER:
        print("Tkinter not available for interactive folder selection.")
        print("Please provide folder path as command line argument.")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(
        title="Select Checkpoint Folder",
        initialdir="/home/jonathan/Data"
    )
    root.update()
    root.quit()
    root.destroy()

    if not folder_path:
        print("No folder selected.")
        sys.exit(1)
    return folder_path


def main():
    if len(sys.argv) == 2:
        folder_path = sys.argv[1]
    elif len(sys.argv) == 1:
        print("No folder specified. Opening interactive folder selection...")
        folder_path = select_folder_interactive()
    else:
        print("Usage: python plot_loss.py [checkpoint_folder]")
        sys.exit(1)

    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        sys.exit(1)

    folder_name = Path(folder_path).name
    print(f"Scanning checkpoint folder: {folder_path}")
    checkpoints = scan_checkpoints(folder_path)

    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoints")
        print("Sample data points (oldest first):")
        for epoch, batch, loss, _m, _fn in checkpoints[:5]:
            print(f"  Epoch {epoch}, Batch {batch}: Loss {loss:.4f}")
        if len(checkpoints) > 5:
            print(f"  ... and {len(checkpoints) - 5} more")
        bpe = max(cp[1] for cp in checkpoints)
        print(f"Using batches-per-epoch scale ≈ {bpe} (from max batch in filenames)")

        try:
            plot_loss_progression(checkpoints, folder_name)
        except Exception as e:
            print(f"Error creating plot: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")
    else:
        print("No checkpoints found in the specified folder.")


if __name__ == "__main__":
    main()
