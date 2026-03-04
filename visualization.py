import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

# Folders
dataset_dir = "Dataset"          # Processed CSVs
output_dir = "Visualizations"    # Where to save plots
os.makedirs(output_dir, exist_ok=True)

# Find all signal CSVs
signal_files = sorted(glob(os.path.join(dataset_dir, "*_signals.csv")))

for sig_file in signal_files:
    participant = os.path.basename(sig_file).split("_")[0]
    
    # Read signals
    signals = pd.read_csv(sig_file, parse_dates=['timestamp'])
    
    # Read events if exists
    events_file = os.path.join(dataset_dir, f"{participant}_events.csv")
    if os.path.exists(events_file):
        events = pd.read_csv(events_file, parse_dates=['start', 'end'])
    else:
        events = pd.DataFrame(columns=['start', 'end', 'value', 'stage'])
    
    # Interpolate missing Flow/Thorac/SPO2
    signals[['Flow', 'Thorac', 'SPO2']] = signals[['Flow', 'Thorac', 'SPO2']].interpolate()

    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(signals['timestamp'], signals['Flow'], label='Flow', color='blue')
    plt.plot(signals['timestamp'], signals['Thorac'], label='Thorac', color='green')
    plt.plot(signals['timestamp'], signals['SPO2'], label='SPO2', color='orange')

    # Overlay events
    for _, row in events.iterrows():
        plt.axvspan(row['start'], row['end'], color='red', alpha=0.3)
    
    plt.title(f"Participant {participant} Signals")
    plt.xlabel("Time")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f"{participant}_visualization.pdf"))
    plt.close()

print("All participant visualizations saved in:", output_dir)