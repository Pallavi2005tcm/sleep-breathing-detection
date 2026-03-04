

# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import argparse

# def read_signal(file_path, value_name):
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"{file_path} does not exist")
    
#     # Read lines manually, ignore headers like "Signal Type" or "Length"
#     data = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith('Signal Type') or line.startswith('Length'):
#                 continue
#             data.append(line)
    
#     # Create dataframe
#     df = pd.DataFrame([x.split(';') for x in data], columns=['timestamp', value_name])
#     df['timestamp'] = df['timestamp'].str.strip()
#     df[value_name] = pd.to_numeric(df[value_name].str.strip(), errors='coerce')
#     df = df.dropna()
    
#     # Convert timestamp
#     try:
#         df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S,%f')
#     except:
#         # If only time is present, add a dummy date
#         df['timestamp'] = pd.to_datetime('2024-01-01 ' + df['timestamp'])
#     return df

# def read_events(file_path):
#     if not os.path.exists(file_path):
#         return None
#     df = pd.read_csv(file_path, sep=';', header=None, names=['time_range', 'duration', 'event', 'stage'])
#     df[['start', 'end']] = df['time_range'].str.split('-', expand=True)
#     df['start'] = pd.to_datetime('2024-01-01 ' + df['start'].str.strip())
#     df['end'] = pd.to_datetime('2024-01-01 ' + df['end'].str.strip())
#     return df

# def read_sleep(file_path):
#     if not os.path.exists(file_path):
#         return None
#     df = pd.read_csv(file_path, sep=';', header=None, names=['timestamp', 'stage'])
#     try:
#         df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S,%f')
#     except:
#         df['timestamp'] = pd.to_datetime('2024-01-01 ' + df['timestamp'])
#     return df

# def plot_signals(participant_folder):
#     # Filenames
#     flow_file = os.path.join(participant_folder, 'Flow.txt')
#     thorac_file = os.path.join(participant_folder, 'Thorac.txt')
#     spo2_file = os.path.join(participant_folder, 'SPO2.txt')
#     sleep_file = os.path.join(participant_folder, 'Sleep_profile.txt')
#     events_file = os.path.join(participant_folder, 'Flow_events.csv')
    
#     # Read signals
#     flow = read_signal(flow_file, 'Flow')
#     thorac = read_signal(thorac_file, 'Thorac')
#     spo2 = read_signal(spo2_file, 'SpO2')
#     sleep = read_sleep(sleep_file)
#     events = read_events(events_file)

#     plt.figure(figsize=(15,10))

#     plt.subplot(3,1,1)
#     plt.plot(flow['timestamp'], flow['Flow'], label='Nasal Airflow', color='blue')
#     if events is not None:
#         for _, row in events.iterrows():
#             plt.axvspan(row['start'], row['end'], color='red', alpha=0.3)
#     plt.title('Nasal Airflow with Events')
#     plt.xlabel('Time')
#     plt.ylabel('Flow')

#     plt.subplot(3,1,2)
#     plt.plot(thorac['timestamp'], thorac['Thorac'], label='Thoracic Movement', color='green')
#     if events is not None:
#         for _, row in events.iterrows():
#             plt.axvspan(row['start'], row['end'], color='red', alpha=0.3)
#     plt.title('Thoracic Movement with Events')
#     plt.xlabel('Time')
#     plt.ylabel('Thorac')

#     plt.subplot(3,1,3)
#     plt.plot(spo2['timestamp'], spo2['SpO2'], label='SpO2', color='orange')
#     if events is not None:
#         for _, row in events.iterrows():
#             plt.axvspan(row['start'], row['end'], color='red', alpha=0.3)
#     plt.title('SpO2 with Events')
#     plt.xlabel('Time')
#     plt.ylabel('SpO2 (%)')

#     plt.tight_layout()
    
#     os.makedirs('Visualizations', exist_ok=True)
#     participant_name = os.path.basename(participant_folder)
#     output_path = os.path.join('Visualizations', f'{participant_name}_visualization.pdf')
#     plt.savefig(output_path)
#     print(f"Saved visualization to {output_path}")
#     plt.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-name', '--participant_folder', type=str, required=True)
#     args = parser.parse_args()
#     plot_signals(args.participant_folder)


#!/usr/bin/env python3
"""
Visualization script for sleep breathing data.
Creates a PDF with Nasal Airflow, Thoracic Movement, and SpO₂ signals
with overlaid breathing events.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def get_base_date_from_file(filepath):
    """Extract base date from the Start Time line in the file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('Start Time:'):
                    # Format: "Start Time: 5/30/2024 8:59:00 PM" or "Start Time: 30-05-2024 21:22:45"
                    time_str = line.split(':', 1)[1].strip()
                    
                    # Try different date formats
                    formats = [
                        '%m/%d/%Y %I:%M:%S %p',  # US: 5/30/2024 8:59:00 PM
                        '%d-%m-%Y %H:%M:%S',     # EU: 30-05-2024 21:22:45
                        '%m/%d/%Y %H:%M:%S',     # US without AM/PM
                        '%d.%m.%Y %H:%M:%S',     # Dot format: 30.05.2024 21:22:45
                    ]
                    
                    for fmt in formats:
                        try:
                            return datetime.strptime(time_str, fmt)
                        except ValueError:
                            continue
                    
                    print(f"Warning: Could not parse date format: {time_str}")
                    return None
    except Exception as e:
        print(f"Warning: Could not read Start Time in {filepath}: {e}")
    return None

def parse_timestamp(timestamp_str, base_date=None):
    """
    Parse timestamp in different formats.
    If base_date is provided and timestamp doesn't have date, use base_date.
    """
    timestamp_str = timestamp_str.strip()
    
    # Try different timestamp formats
    formats = [
        '%d.%m.%Y %H:%M:%S,%f',  # DD.MM.YYYY HH:MM:SS,mmm
        '%d.%m.%Y %H:%M:%S',      # DD.MM.YYYY HH:MM:SS
        '%d-%m-%Y %H:%M:%S',      # DD-MM-YYYY HH:MM:SS
        '%m/%d/%Y %H:%M:%S',      # MM/DD/YYYY HH:MM:SS
    ]
    
    # Try formats with full date
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    
    # If base_date is provided, try time-only formats
    if base_date is not None:
        try:
            # Try time format with microseconds (HH:MM:SS,mmm)
            if ',' in timestamp_str:
                time_part, ms_part = timestamp_str.split(',')
                time_obj = datetime.strptime(time_part, '%H:%M:%S')
                dt = datetime.combine(base_date.date(), time_obj.time())
                dt = dt.replace(microsecond=int(ms_part) * 1000)
                return dt
        except:
            try:
                # Try time format without microseconds (HH:MM:SS)
                time_obj = datetime.strptime(timestamp_str, '%H:%M:%S')
                return datetime.combine(base_date.date(), time_obj.time())
            except:
                pass
    
    raise ValueError(f"Could not parse timestamp: {timestamp_str}")

def read_signal_file(filepath):
    """
    Read signal file with header and data.
    Returns DataFrame with datetime index and values.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found")
        return None
    
    # Get base date from file header
    base_date = get_base_date_from_file(filepath)
    if base_date is None:
        print(f"Warning: Could not find Start Time in {filepath}")
        return None
    
    # Read file, skipping header lines
    data_lines = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
        # Find where data starts (after "Data:" line)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('Data:'):
                data_start = i + 1
                break
        
        # If no "Data:" line found, look for first line with semicolon
        if data_start == 0:
            for i, line in enumerate(lines):
                if ';' in line and not line.startswith('Signal') and not line.startswith('Start'):
                    data_start = i
                    break
        
        # Read data lines
        for line in lines[data_start:]:
            if ';' in line and line.strip():
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    timestamp_str = parts[0].strip()
                    try:
                        value = float(parts[1].strip().replace(',', '.'))
                    except:
                        continue
                    
                    try:
                        dt = parse_timestamp(timestamp_str, base_date)
                        data_lines.append([dt, value])
                    except Exception as e:
                        # Skip unparseable timestamps
                        continue
    
    if not data_lines:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data_lines, columns=['datetime', 'value'])
    df.set_index('datetime', inplace=True)
    
    # Sort index just in case
    df.sort_index(inplace=True)
    
    # Remove duplicates (keep first)
    df = df[~df.index.duplicated(keep='first')]
    
    return df

def read_events_file(filepath):
    """
    Read flow events file.
    Returns list of events with start, end, type, and stage.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Events file {filepath} not found")
        return []
    
    # Get base date from the file
    base_date = get_base_date_from_file(filepath)
    if base_date is None:
        print(f"Warning: Could not find Start Time in {filepath}")
        return []
    
    events = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
        # Find where data starts
        data_start = 0
        for i, line in enumerate(lines):
            if ';' in line and not line.startswith('Signal') and not line.startswith('Start'):
                data_start = i
                break
        
        # Read event lines
        for line in lines[data_start:]:
            if ';' in line and line.strip():
                parts = line.strip().split(';')
                if len(parts) >= 4:
                    # Parse time range
                    time_range = parts[0].strip()
                    if '-' in time_range:
                        start_str, end_str = time_range.split('-')
                        
                        try:
                            # Parse start and end times
                            start_time = parse_timestamp(start_str, base_date)
                            end_time = parse_timestamp(end_str, base_date)
                            
                            # Handle case where end time might be on next day
                            if end_time < start_time:
                                end_time += timedelta(days=1)
                            
                            duration = int(parts[1].strip())
                            event_type = parts[2].strip()
                            sleep_stage = parts[3].strip() if len(parts) > 3 else 'Unknown'
                            
                            events.append({
                                'start': start_time,
                                'end': end_time,
                                'duration': duration,
                                'type': event_type,
                                'stage': sleep_stage
                            })
                        except Exception as e:
                            # Skip unparseable events
                            continue
    
    return events

def create_visualization(participant_folder, output_dir='Visualizations'):
    """
    Create visualization PDF for a participant.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get participant name from folder path
    participant = os.path.basename(participant_folder)
    
    # Define file paths
    nasal_file = os.path.join(participant_folder, 'Flow.txt')
    thoracic_file = os.path.join(participant_folder, 'Thorac.txt')
    spo2_file = os.path.join(participant_folder, 'SPO2.txt')
    events_file = os.path.join(participant_folder, 'Flow Events.txt')
    
    # Check if files exist
    for file in [nasal_file, thoracic_file, spo2_file, events_file]:
        if not os.path.exists(file):
            print(f"Warning: File {file} does not exist")
    
    # Read signals
    print(f"Reading signals for {participant}...")
    nasal_df = read_signal_file(nasal_file)
    thoracic_df = read_signal_file(thoracic_file)
    spo2_df = read_signal_file(spo2_file)
    
    if nasal_df is None:
        print(f"Error: Could not read nasal airflow file for {participant}")
        return
    if thoracic_df is None:
        print(f"Error: Could not read thoracic movement file for {participant}")
        return
    if spo2_df is None:
        print(f"Error: Could not read SpO2 file for {participant}")
        return
    
    # Read events
    events = read_events_file(events_file)
    print(f"Found {len(events)} events")
    
    # Print some basic info
    print(f"Nasal airflow: {len(nasal_df)} samples from {nasal_df.index[0]} to {nasal_df.index[-1]}")
    print(f"Thoracic movement: {len(thoracic_df)} samples from {thoracic_df.index[0]} to {thoracic_df.index[-1]}")
    print(f"SpO2: {len(spo2_df)} samples from {spo2_df.index[0]} to {spo2_df.index[-1]}")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    
    # Plot Nasal Airflow
    axes[0].plot(nasal_df.index, nasal_df['value'], 'b-', linewidth=0.5, label='Nasal Airflow')
    axes[0].set_ylabel('Nasal Airflow (a.u.)')
    axes[0].set_title(f'{participant} - Nasal Airflow (32 Hz)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Thoracic Movement
    axes[1].plot(thoracic_df.index, thoracic_df['value'], 'g-', linewidth=0.5, label='Thoracic Movement')
    axes[1].set_ylabel('Thoracic Movement (a.u.)')
    axes[1].set_title(f'{participant} - Thoracic Movement (32 Hz)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot SpO₂
    axes[2].plot(spo2_df.index, spo2_df['value'], 'r-', linewidth=1, label='SpO₂')
    axes[2].set_ylabel('SpO₂ (%)')
    axes[2].set_xlabel('Time')
    axes[2].set_title(f'{participant} - Oxygen Saturation (4 Hz)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # Overlay events on all subplots
    colors = {'Hypopnea': 'yellow', 'Obstructive Apnea': 'red'}
    for event in events:
        for ax in axes:
            ax.axvspan(event['start'], event['end'], 
                      alpha=0.3, color=colors.get(event['type'], 'gray'),
                      label=event['type'] if event == events[0] else "")
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=1))
    
    # Add legend for events only once
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc='upper left')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save to PDF
    output_file = os.path.join(output_dir, f'{participant}_visualization.pdf')
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_file}")
    
    # Also save a sample plot for first 5 minutes to see details
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    
    # Get first 5 minutes of data
    end_time = nasal_df.index[0] + timedelta(minutes=5)
    
    nasal_sample = nasal_df.loc[:end_time]
    thoracic_sample = thoracic_df.loc[:end_time]
    spo2_sample = spo2_df.loc[:end_time]
    
    axes[0].plot(nasal_sample.index, nasal_sample['value'], 'b-', linewidth=1)
    axes[0].set_ylabel('Nasal Airflow')
    axes[0].set_title(f'{participant} - First 5 minutes (Nasal Airflow)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(thoracic_sample.index, thoracic_sample['value'], 'g-', linewidth=1)
    axes[1].set_ylabel('Thoracic Movement')
    axes[1].set_title('First 5 minutes (Thoracic Movement)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(spo2_sample.index, spo2_sample['value'], 'r-', linewidth=1)
    axes[2].set_ylabel('SpO₂ (%)')
    axes[2].set_xlabel('Time')
    axes[2].set_title('First 5 minutes (SpO₂)')
    axes[2].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    sample_file = os.path.join(output_dir, f'{participant}_sample_5min.pdf')
    plt.savefig(sample_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sample visualization (5 minutes) saved to {sample_file}")

def main():
    parser = argparse.ArgumentParser(description='Create visualization of sleep breathing data')
    parser.add_argument('-name', type=str, required=True, 
                       help='Path to participant folder (e.g., Data/AP20)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.name):
        print(f"Error: Participant folder {args.name} does not exist")
        return
    
    create_visualization(args.name)

if __name__ == '__main__':
    main()