


# import os
# import pandas as pd
# from tqdm import tqdm

# def read_signal(file_path, col_name, sample_rate=None):
#     """
#     Reads a signal file (Flow, Thorac, SpO2) and returns a DataFrame
#     with timestamp and signal column.
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"{file_path} does not exist")
    
#     df = pd.read_csv(file_path)
    
#     # Ensure timestamp column exists
#     if 'timestamp' not in df.columns:
#         raise ValueError(f"{file_path} missing 'timestamp' column")
    
#     # Convert timestamp to datetime
#     df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True)
    
#     # Rename signal column
#     if col_name not in df.columns:
#         # Some files may only have values in first column after timestamp
#         df[col_name] = df.iloc[:, 1]
    
#     df = df[['timestamp', col_name]].copy()
    
#     # Resample/interpolate if sample_rate is given
#     if sample_rate:
#         df.set_index('timestamp', inplace=True)
#         # Calculate period in milliseconds
#         ms = int(1000 / sample_rate)
#         df = df.resample(f'{ms}ms').interpolate()
#         df.reset_index(inplace=True)
    
#     return df

# def main(in_dir, out_dir):
#     os.makedirs(out_dir, exist_ok=True)
    
#     participants = [p for p in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, p))]
    
#     for p in tqdm(participants, desc="Processing participants"):
#         folder = os.path.join(in_dir, p)
#         try:
#             # Read all signals
#             flow   = read_signal(os.path.join(folder, 'Flow.txt'), 'Flow', sample_rate=32)
#             thorac = read_signal(os.path.join(folder, 'Thorac.txt'), 'Thorac', sample_rate=32)
#             spo2   = read_signal(os.path.join(folder, 'SPO2.txt'), 'SpO2', sample_rate=4)
            
#             # Merge all signals on timestamp
#             signals = pd.merge_asof(flow, thorac, on='timestamp')
#             signals = pd.merge_asof(signals, spo2, on='timestamp')
            
#             # Save dataset CSV
#             out_file = os.path.join(out_dir, f"{p}_dataset.csv")
#             signals.to_csv(out_file, index=False)
#         except Exception as e:
#             print(f"Skipping {p} due to error: {e}")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-in_dir", required=True, help="Input data folder with participant subfolders")
#     parser.add_argument("-out_dir", required=True, help="Output folder to save datasets")
#     args = parser.parse_args()
    
#     main(args.in_dir, args.out_dir)

#!/usr/bin/env python3
"""
Dataset creation script for sleep breathing data.
Filters signals, creates 30-second windows with 50% overlap,
and assigns labels based on events.
"""

import argparse
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import os
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

class BreathingDatasetCreator:
    def __init__(self):
        self.window_size = 30  # seconds
        self.overlap = 0.5  # 50% overlap
        self.sampling_rates = {
            'nasal': 32,
            'thoracic': 32,
            'spo2': 4
        }
        
    def get_base_date_from_file(self, filepath):
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
    
    def parse_timestamp(self, timestamp_str, base_date=None):
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
    
    def read_signal_file(self, filepath):
        """Read signal file with header and data."""
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found")
            return None
        
        # Get base date from file header
        base_date = self.get_base_date_from_file(filepath)
        if base_date is None:
            print(f"Warning: Could not find Start Time in {filepath}")
            return None
        
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
                            dt = self.parse_timestamp(timestamp_str, base_date)
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
    
    def read_events_file(self, filepath):
        """Read flow events file."""
        if not os.path.exists(filepath):
            print(f"Warning: Events file {filepath} not found")
            return []
        
        # Get base date from the file
        base_date = self.get_base_date_from_file(filepath)
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
                                start_time = self.parse_timestamp(start_str, base_date)
                                end_time = self.parse_timestamp(end_str, base_date)
                                
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
    
    def read_sleep_profile(self, filepath):
        """Read sleep profile file."""
        if not os.path.exists(filepath):
            print(f"Warning: Sleep profile file {filepath} not found")
            return None
        
        # Get base date from the file
        base_date = self.get_base_date_from_file(filepath)
        if base_date is None:
            print(f"Warning: Could not find Start Time in {filepath}")
            return None
        
        data_lines = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
            # Find where data starts
            data_start = 0
            for i, line in enumerate(lines):
                if ';' in line and not line.startswith('Signal') and not line.startswith('Start') and not line.startswith('Events'):
                    data_start = i
                    break
            
            # Read data lines
            for line in lines[data_start:]:
                if ';' in line and line.strip():
                    parts = line.strip().split(';')
                    if len(parts) >= 2:
                        timestamp_str = parts[0].strip()
                        stage = parts[1].strip()
                        
                        try:
                            dt = self.parse_timestamp(timestamp_str, base_date)
                            data_lines.append([dt, stage])
                        except Exception as e:
                            continue
        
        if not data_lines:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data_lines, columns=['datetime', 'stage'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        """Create Butterworth bandpass filter."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def bandpass_filter(self, data, fs, lowcut=0.17, highcut=0.4, order=4):
        """Apply bandpass filter to retain breathing frequency range (0.17-0.4 Hz)."""
        if len(data) < 10:  # Too short for filtering
            return data
        
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        
        # Handle NaN values
        try:
            # Convert to numpy array
            data_array = data.values.copy()
            
            # Handle NaN values by linear interpolation
            if np.isnan(data_array).any():
                # Create a pandas Series for interpolation
                data_series = pd.Series(data_array, index=data.index)
                # Interpolate NaN values
                data_series = data_series.interpolate(method='linear', limit_direction='both')
                data_array = data_series.values
            
            # Apply filter
            filtered = filtfilt(b, a, data_array)
            return pd.Series(filtered, index=data.index)
        except Exception as e:
            print(f"Warning: Filtering failed: {e}")
            return data
    
    def create_windows(self, df, fs, start_time, end_time):
        """
        Create 30-second windows with 50% overlap from a signal DataFrame.
        Returns list of (window_start, window_data) tuples.
        """
        windows = []
        window_samples = int(self.window_size * fs)
        step_samples = int(window_samples * (1 - self.overlap))
        
        current_time = start_time
        
        while current_time + timedelta(seconds=self.window_size) <= end_time:
            window_end = current_time + timedelta(seconds=self.window_size)
            
            # Get data for this window
            window_data = df.loc[current_time:window_end].values.flatten()
            
            # Pad if necessary
            if len(window_data) < window_samples:
                padding = np.full(window_samples - len(window_data), np.nan)
                window_data = np.concatenate([window_data, padding])
            elif len(window_data) > window_samples:
                window_data = window_data[:window_samples]
            
            windows.append({
                'start_time': current_time,
                'end_time': window_end,
                'data': window_data
            })
            
            current_time += timedelta(seconds=step_samples)
        
        return windows
    
    def get_window_label(self, window_start, window_end, events, overlap_threshold=0.5):
        """
        Determine label for a window based on event overlap.
        Returns event type if overlap > threshold, else 'Normal'.
        """
        window_duration = (window_end - window_start).total_seconds()
        max_overlap = 0
        assigned_label = 'Normal'
        
        for event in events:
            # Calculate overlap
            overlap_start = max(window_start, event['start'])
            overlap_end = min(window_end, event['end'])
            
            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                overlap_ratio = overlap_duration / window_duration
                
                if overlap_ratio > max_overlap and overlap_ratio > overlap_threshold:
                    max_overlap = overlap_ratio
                    assigned_label = event['type']
        
        return assigned_label
    
    def process_participant(self, participant_folder):
        """Process all data for a single participant."""
        participant = os.path.basename(participant_folder)
        print(f"\nProcessing {participant}...")
        
        # Define file paths
        nasal_file = os.path.join(participant_folder, 'Flow.txt')
        thoracic_file = os.path.join(participant_folder, 'Thorac.txt')
        spo2_file = os.path.join(participant_folder, 'SPO2.txt')
        events_file = os.path.join(participant_folder, 'Flow Events.txt')
        sleep_file = os.path.join(participant_folder, 'Sleep profile.txt')
        
        # Read signals
        print(f"  Reading signal files...")
        nasal_df = self.read_signal_file(nasal_file)
        thoracic_df = self.read_signal_file(thoracic_file)
        spo2_df = self.read_signal_file(spo2_file)
        
        if nasal_df is None:
            print(f"  Error: Could not read nasal airflow file")
            return None, None
        if thoracic_df is None:
            print(f"  Error: Could not read thoracic movement file")
            return None, None
        if spo2_df is None:
            print(f"  Error: Could not read SpO2 file")
            return None, None
        
        # Read events and sleep profile
        print(f"  Reading events and sleep profile...")
        events = self.read_events_file(events_file)
        sleep_df = self.read_sleep_profile(sleep_file)
        
        print(f"  Found {len(events)} events")
        
        # Apply bandpass filter to respiratory signals (0.17-0.4 Hz)
        print(f"  Applying bandpass filter (0.17-0.4 Hz)...")
        nasal_df['filtered'] = self.bandpass_filter(nasal_df['value'], self.sampling_rates['nasal'])
        thoracic_df['filtered'] = self.bandpass_filter(thoracic_df['value'], self.sampling_rates['thoracic'])
        
        # Get overall time range
        start_time = min(nasal_df.index[0], thoracic_df.index[0], spo2_df.index[0])
        end_time = max(nasal_df.index[-1], thoracic_df.index[-1], spo2_df.index[-1])
        
        print(f"  Time range: {start_time} to {end_time}")
        
        # Create windows for each signal
        print(f"  Creating windows ({self.window_size}s with {self.overlap*100}% overlap)...")
        nasal_windows = self.create_windows(nasal_df['filtered'], self.sampling_rates['nasal'], start_time, end_time)
        thoracic_windows = self.create_windows(thoracic_df['filtered'], self.sampling_rates['thoracic'], start_time, end_time)
        spo2_windows = self.create_windows(spo2_df['value'], self.sampling_rates['spo2'], start_time, end_time)
        
        print(f"  Created {len(nasal_windows)} windows")
        
        # Align windows (use nasal windows as reference)
        all_windows = []
        labels = []
        
        for i, nasal_win in enumerate(nasal_windows):
            if i < len(thoracic_windows) and i < len(spo2_windows):
                # Get label based on events
                label = self.get_window_label(
                    nasal_win['start_time'], 
                    nasal_win['end_time'], 
                    events
                )
                
                # Get sleep stage for this window (if available)
                sleep_stage = 'Unknown'
                if sleep_df is not None:
                    # Find the sleep stage that covers this window
                    stage_rows = sleep_df.loc[
                        (sleep_df.index >= nasal_win['start_time']) & 
                        (sleep_df.index <= nasal_win['end_time'])
                    ]
                    if not stage_rows.empty:
                        sleep_stage = stage_rows.iloc[0]['stage']
                
                all_windows.append({
                    'participant': participant,
                    'window_id': i,
                    'start_time': nasal_win['start_time'],
                    'end_time': nasal_win['end_time'],
                    'nasal_flow': nasal_win['data'],
                    'thoracic': thoracic_windows[i]['data'],
                    'spo2': spo2_windows[i]['data'],
                    'label': label,
                    'sleep_stage': sleep_stage
                })
                
                labels.append(label)
        
        print(f"  Successfully created {len(all_windows)} aligned windows")
        print(f"  Label distribution: {pd.Series(labels).value_counts().to_dict()}")
        
        return all_windows, sleep_df
    
    def process_all_participants(self, input_dir, output_dir):
        """Process all participants in the input directory."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        all_breathing_data = []
        all_sleep_data = []
        
        # Find all participant folders (AP01, AP02, etc.)
        participant_folders = [d for d in os.listdir(input_dir) 
                              if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('AP')]
        
        print(f"Found {len(participant_folders)} participant folders: {participant_folders}")
        
        successful_participants = 0
        for participant_folder in sorted(participant_folders):
            full_path = os.path.join(input_dir, participant_folder)
            breathing_windows, sleep_df = self.process_participant(full_path)
            
            if breathing_windows:
                all_breathing_data.extend(breathing_windows)
                successful_participants += 1
            
            if sleep_df is not None:
                sleep_df['participant'] = participant_folder
                all_sleep_data.append(sleep_df)
        
        print(f"\nSuccessfully processed {successful_participants} out of {len(participant_folders)} participants")
        
        # Save breathing dataset
        if all_breathing_data:
            print(f"\nSaving datasets...")
            
            # Convert to DataFrame-friendly format for CSV
            df_breathing = pd.DataFrame([
                {
                    'participant': w['participant'],
                    'window_id': w['window_id'],
                    'start_time': w['start_time'],
                    'end_time': w['end_time'],
                    'label': w['label'],
                    'sleep_stage': w['sleep_stage'],
                    'nasal_flow_mean': float(np.mean(w['nasal_flow'])),
                    'nasal_flow_std': float(np.std(w['nasal_flow'])),
                    'thoracic_mean': float(np.mean(w['thoracic'])),
                    'thoracic_std': float(np.std(w['thoracic'])),
                    'spo2_mean': float(np.mean(w['spo2'])),
                    'spo2_min': float(np.min(w['spo2'])),
                    'spo2_std': float(np.std(w['spo2']))
                }
                for w in all_breathing_data
            ])
            
            # Save to CSV
            csv_path = os.path.join(output_dir, 'breathing_dataset.csv')
            df_breathing.to_csv(csv_path, index=False)
            print(f"  Saved breathing dataset (CSV) to {csv_path}")
            print(f"  Total windows: {len(df_breathing)}")
            print(f"  Label distribution:\n{df_breathing['label'].value_counts()}")
            
            # Also save full data with signals as pickle (for modeling)
            pickle_path = os.path.join(output_dir, 'breathing_dataset_full.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(all_breathing_data, f)
            print(f"  Saved full dataset with signals to {pickle_path}")
        
        # Save sleep dataset
        if all_sleep_data:
            df_sleep = pd.concat(all_sleep_data)
            df_sleep.to_csv(os.path.join(output_dir, 'sleep_stage_dataset.csv'))
            print(f"  Saved sleep dataset to {os.path.join(output_dir, 'sleep_stage_dataset.csv')}")
        
        return all_breathing_data

def main():
    parser = argparse.ArgumentParser(description='Create dataset from sleep breathing data')
    parser.add_argument('-in_dir', type=str, required=True, help='Input directory containing participant folders')
    parser.add_argument('-out_dir', type=str, required=True, help='Output directory for dataset')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.in_dir):
        print(f"Error: Input directory {args.in_dir} does not exist")
        return
    
    print(f"Input directory: {args.in_dir}")
    print(f"Output directory: {args.out_dir}")
    
    creator = BreathingDatasetCreator()
    creator.process_all_participants(args.in_dir, args.out_dir)
    
    print("\nDataset creation complete!")

if __name__ == '__main__':
    main()