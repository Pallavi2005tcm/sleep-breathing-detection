# Sleep Breathing Irregularity Detection

## Project Overview

This project analyzes overnight sleep data to detect breathing irregularities (Apnea and Hypopnea) using physiological signals collected from 5 participants. The goal is to preprocess multi-rate time-series data, generate labeled windows, and train a model to classify abnormal breathing events.

## Dataset

Each participant folder contains:

- **Flow.txt** – Nasal airflow signal (32 Hz)  
- **Thorac.txt** – Thoracic movement signal (32 Hz)  
- **SPO2.txt** – Oxygen saturation signal (4 Hz)  
- **Flow Events.txt** – Annotations for breathing events  
- **Sleep profile.txt** – Sleep stages recorded at 30-second intervals  

## Project Components

### 1. Visualization (`vis.py`)

- Aligns multi-rate time-series signals using timestamps  
- Plots full 8-hour recordings  
- Overlays annotated breathing events  
- Saves visualization as a PDF in the `Visualizations` directory  

**Usage:**

```bash
python vis.py -name "Data/AP20"
```

### 2. Dataset Creation (`create_dataset.py`)

- Applies bandpass filtering (0.17–0.4 Hz) to retain breathing frequencies  
- Segments signals into 30-second windows with 50% overlap  
- Labels windows based on >50% overlap with annotated events  
- Saves processed dataset to the `Dataset` directory  

**Usage:**

```bash
python create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

### 3. Modeling & Evaluation

- Trains a 1D CNN model for breathing irregularity classification  
- Uses Leave-One-Participant-Out Cross-Validation  
- Reports Accuracy, Precision, Recall, and Confusion Matrix  

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## AI Tool Usage Disclosure

This project was initially developed with the assistance of DeepSeek. I have since thoroughly reviewed, studied, and understood the complete implementation, including preprocessing, filtering, windowing, labeling logic, model architecture, and evaluation strategy. I am fully comfortable explaining all design decisions and code components during discussion.
