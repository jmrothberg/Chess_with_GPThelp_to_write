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
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend explicitly
import matplotlib.pyplot as plt
import sys
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
    # Match pattern: E{epoch}B{batch}_L{loss}_{timestamp}.pth
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
    Returns list of (epoch, batch, loss) tuples sorted by epoch and batch.
    """
    checkpoints = []

    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return []

    for filename in os.listdir(folder_path):
        if filename.endswith('.pth'):
            data = parse_checkpoint_filename(filename)
            if data:
                checkpoints.append(data)

    # Sort by epoch, then by batch
    checkpoints.sort(key=lambda x: (x[0], x[1]))

    return checkpoints

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

    # Create combined x-axis (epoch + batch fraction)
    x_values = []
    for epoch, batch in zip(epochs, batches):
        # Normalize batch to 0-1 range (assuming ~60k batches per epoch based on your training)
        batch_fraction = batch / 60000.0  # Adjust this based on your actual batch count per epoch
        x_values.append(epoch + batch_fraction)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot loss over time
    plt.subplot(1, 2, 1)
    plt.plot(x_values, losses, 'b-', marker='o', markersize=3, linewidth=1)
    plt.xlabel('Epoch (with batch progression)')
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
    plt.xlabel('Epoch (with batch progression)')
    plt.ylabel('Smoothed Loss (50-point average)')
    plt.title(f'Smoothed Loss Progression - {folder_name}')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def select_folder_interactive():
    """Select folder using tkinter dialog if available."""
    if not HAS_TKINTER:
        print("Tkinter not available for interactive folder selection.")
        print("Please provide folder path as command line argument.")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(
        title="Select Checkpoint Folder",
        initialdir="/data"
    )
    root.update()  # Force update to ensure dialog closes
    root.quit()  # Properly close the tkinter main loop
    root.destroy()  # Clean up the window

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
        print("If no folder is specified, interactive selection will be used.")
        sys.exit(1)

    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        sys.exit(1)

    folder_name = Path(folder_path).name

    print(f"Scanning checkpoint folder: {folder_path}")
    checkpoints = scan_checkpoints(folder_path)

    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoints")
        print("Sample data points:")
        for i, (epoch, batch, loss) in enumerate(checkpoints[:5]):
            print(f"  Epoch {epoch}, Batch {batch}: Loss {loss:.4f}")
        if len(checkpoints) > 5:
            print(f"  ... and {len(checkpoints) - 5} more")

        try:
            plot_loss_progression(checkpoints, folder_name)
        except Exception as e:
            print(f"Error creating plot: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")
    else:
        print("No checkpoints found in the specified folder.")
        print("Make sure the folder contains .pth checkpoint files.")

if __name__ == "__main__":
    main()
