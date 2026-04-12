#!/usr/bin/env python3
"""
Simple loss plotting utility for ChessBrain checkpoints.
Scans a checkpoint folder and plots loss progression over time.

Usage:
  python plot_loss.py <checkpoint_folder>
  python plot_loss.py  # Interactive folder selection
"""

import os
import re
import matplotlib.pyplot as plt
import sys
import json
from pathlib import Path

# For interactive folder selection (if tkinter is available)
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

def parse_checkpoint_filename(filename):
    """
    Parse checkpoint filename to extract epoch, batch, and loss.
    Filename format: C12H12E768_B64_E1B57300_L2.874_1010_2245.pth
    """
    pattern = r'E(\d+)B(\d+)_L([\d.]+)_'
    match = re.search(pattern, filename)

    if match:
        epoch = int(match.group(1))
        batch = int(match.group(2))
        loss = float(match.group(3))
        return epoch, batch, loss
    return None

def scan_checkpoints(folder_path):
    """
    Scan checkpoint folder and extract loss data from filenames.
    Order: oldest file modification time first, then newest (os.path.getmtime).
    Each row: (epoch, batch, loss, mtime, filename).
    """
    checkpoints = []

    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return []

    for filename in os.listdir(folder_path):
        if filename.endswith('.pth'):
            data = parse_checkpoint_filename(filename)
            if data:
                epoch, batch, loss = data
                full_path = os.path.join(folder_path, filename)
                mtime = os.path.getmtime(full_path)
                checkpoints.append((epoch, batch, loss, mtime, filename))

    # Oldest modified → newest
    checkpoints.sort(key=lambda x: (x[3], x[4]))

    return checkpoints

def get_plot_data_filename(folder_path):
    """
    Get the filename for storing plot data in the checkpoint folder.
    Returns the full path to plot_data.json in the folder.
    """
    return os.path.join(folder_path, "plot_data.json")

def load_saved_plot_data(folder_path):
    """
    Load previously saved plot data from the folder.
    Returns list of (epoch, batch, loss, mtime, filename). Legacy rows may lack mtime (0.0).
    """
    data_file = get_plot_data_filename(folder_path)
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
                out = []
                for item in data:
                    epoch = item['epoch']
                    batch = item['batch']
                    loss = item['loss']
                    mtime = float(item.get('mtime', 0.0))
                    fn = item.get('filename', '')
                    out.append((epoch, batch, loss, mtime, fn))
                return out
        except (json.JSONDecodeError, KeyError):
            print(f"Warning: Could not load saved plot data from {data_file}")
            return []
    return []

def save_plot_data(folder_path, checkpoints):
    """
    Save the current checkpoint data to a JSON file in the folder.
    This allows us to keep plot data even after deleting checkpoint files.
    """
    data_file = get_plot_data_filename(folder_path)

    # mtime + filename so order survives after files are deleted
    data = []
    for epoch, batch, loss, mtime, filename in checkpoints:
        row = {
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'mtime': mtime,
            'filename': filename,
        }
        data.append(row)

    try:
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved plot data for {len(checkpoints)} checkpoints to {data_file}")
    except Exception as e:
        print(f"Warning: Could not save plot data to {data_file}: {e}")

def _merge_key(cp):
    """Unique id: filename if present, else (epoch, batch) for legacy rows."""
    epoch, batch, loss, mtime, filename = cp
    if filename:
        return ('file', filename)
    return ('eb', epoch, batch)


def merge_checkpoint_data(current_checkpoints, saved_checkpoints):
    """
    Merge current checkpoint data with saved historical data.
    Current files win on duplicate filename; final order is by mtime (oldest first).
    """
    if not saved_checkpoints:
        return current_checkpoints

    merged = list(saved_checkpoints)
    index_by_key = {_merge_key(cp): i for i, cp in enumerate(merged)}

    for cp in current_checkpoints:
        key = _merge_key(cp)
        existing_index = index_by_key.get(key)
        if existing_index is not None:
            merged[existing_index] = cp
        else:
            index_by_key[key] = len(merged)
            merged.append(cp)

    merged.sort(key=lambda x: (x[3], x[4]))

    return merged

def scan_checkpoints_with_history(folder_path):
    """
    Scan checkpoint folder and merge with any saved historical data.
    This ensures complete plotting data even when checkpoint files are deleted.
    """
    # Get current checkpoints from files
    current_checkpoints = scan_checkpoints(folder_path)

    # Load any saved historical data
    saved_checkpoints = load_saved_plot_data(folder_path)

    # Merge current and saved data
    merged_checkpoints = merge_checkpoint_data(current_checkpoints, saved_checkpoints)

    if saved_checkpoints:
        print(f"Loaded {len(saved_checkpoints)} saved data points")
        if current_checkpoints:
            print(f"Found {len(current_checkpoints)} current checkpoints")
            print(f"Merged into {len(merged_checkpoints)} total data points")
        else:
            print("Using saved data only (no current checkpoints found)")

    return merged_checkpoints

def plot_loss_progression(checkpoints, folder_name):
    """
    Plot loss progression from checkpoint data.
    """
    if not checkpoints:
        print("No valid checkpoints found!")
        return

    # Extract data for plotting
    epochs = [cp[0] for cp in checkpoints]
    batches = [cp[1] for cp in checkpoints]
    losses = [cp[2] for cp in checkpoints]

    # Create combined x-axis that never backtracks
    # Track cumulative batch count across all epochs
    x_values = []
    cumulative_batch = 0
    last_epoch = epochs[0]
    
    for epoch, batch in zip(epochs, batches):
        # When epoch changes, we continue from where we left off (no reset)
        # Just use the current batch number as offset for this epoch
        if epoch != last_epoch:
            # Epoch changed - this is expected, just continue forward
            last_epoch = epoch
        
        # Progressive x-axis: each checkpoint gets the next sequential position
        x_values.append(cumulative_batch / 1000.0)  # Scale for readability
        cumulative_batch += 1

    # Create the plot
    
    plt.figure(figsize=(12, 6))

    # Plot loss over time
    plt.subplot(1, 2, 1)
    plt.plot(x_values, losses, 'b-', marker='o', markersize=3, linewidth=1)
    plt.xlabel('Training Progress (x1000 checkpoints)')
    plt.ylabel('Loss')
    plt.title(f'Loss Progression - {folder_name}')
    plt.grid(True, alpha=0.3)

    # Plot smoothed loss with sliding window average of last 50 losses
    plt.subplot(1, 2, 2)

    # Calculate sliding window average with window size 50
    window_size = 50
    smoothed_losses = []
    smoothed_x = []

    for i in range(len(losses)):
        if i >= window_size - 1:
            # Calculate average of last 50 losses
            window_avg = sum(losses[i-window_size+1:i+1]) / window_size
            smoothed_losses.append(window_avg)
            smoothed_x.append(x_values[i])

    # Plot the smoothed losses as red line
    plt.plot(smoothed_x, smoothed_losses, 'r-', linewidth=2)
    plt.xlabel('Training Progress (x1000 checkpoints)')
    plt.ylabel('Smoothed Loss (50-point average)')
    plt.title(f'Smoothed Loss Progression - {folder_name}')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def select_folder_interactive(initialdir=None):
    """Select folder using tkinter dialog if available."""
    if not HAS_TKINTER:
        print("Tkinter not available for interactive folder selection.")
        print("Please provide folder path as command line argument.")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(
        title="Select Checkpoint Folder",
        initialdir=initialdir or "/data"
    )
    root.destroy()

    if not folder_path:
        print("No folder selected.")
        sys.exit(1)

    return folder_path

def main():
    import platform

    # Determine the default starting directory based on platform
    if platform.system() == "Darwin":
        default_start_dir = "/Users/jonathanrothberg/Data"
    else:
        default_start_dir = "/data/Data"

    # Command line argument specifies starting directory for interactive selection
    start_dir = sys.argv[1] if len(sys.argv) == 2 else default_start_dir

    print("Opening interactive folder selection...")
    folder_path = select_folder_interactive(start_dir)

    # If the folder doesn't exist, prompt for interactive selection
    while not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist. Please select a valid folder.")
        folder_path = select_folder_interactive(start_dir)

    folder_name = Path(folder_path).name

    print(f"Scanning checkpoint folder: {folder_path}")
    checkpoints = scan_checkpoints_with_history(folder_path)

    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoints")
        print("Sample data points:")
        for i, (epoch, batch, loss, _mtime, _fn) in enumerate(checkpoints[:5]):
            print(f"  Epoch {epoch}, Batch {batch}: Loss {loss:.4f}")
        if len(checkpoints) > 5:
            print(f"  ... and {len(checkpoints) - 5} more")

        # Display the lowest 5 scores
        print("\nLowest 5 scores:")
        lowest_scores = sorted(checkpoints, key=lambda x: x[2])[:5]
        for i, (epoch, batch, loss, _m, _fn) in enumerate(lowest_scores, 1):
            print(f"  {i}. Epoch {epoch}, Batch {batch}: Loss {loss:.4f}")

        try:
            # Save the merged checkpoint data before plotting
            # This ensures data is saved even if user closes plot window prematurely
            print(f"Saving plot data for {len(checkpoints)} checkpoints...")
            save_plot_data(folder_path, checkpoints)

            plot_loss_progression(checkpoints, folder_name)

        except Exception as e:
            print(f"Error creating plot: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")
    else:
        print("No checkpoints found in the specified folder.")
        print("Make sure the folder contains .pth checkpoint files.")

if __name__ == "__main__":
    main()
