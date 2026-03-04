# Sleep Breathing Irregularity Detection

## Project Overview
This project analyzes overnight sleep data to detect breathing irregularities (apnea and hypopnea) using physiological signals from 5 participants.

## Dataset
Each participant folder contains:
- `Flow.txt`: Nasal airflow signal (32 Hz)
- `Thorac.txt`: Thoracic movement signal (32 Hz)
- `SPO2.txt`: Oxygen saturation signal (4 Hz)
- `Flow Events.txt`: Annotations for breathing events
- `Sleep profile.txt`: Sleep stages (30s intervals)

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt