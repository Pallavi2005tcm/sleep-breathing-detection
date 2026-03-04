#!/usr/bin/env python3
"""
Training script for breathing irregularity classification.
Trains a 1D CNN model using Leave-One-Participant-Out Cross-Validation.
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class BreathingClassifier:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.class_names = []
        self.models = {}
        self.histories = {}
        
    def build_cnn_model(self, input_shape, num_classes):
        """Build a simple 1D CNN model."""
        model = keras.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape, padding='same'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def prepare_data(self, data_windows, fit_scaler=True):
        """Prepare data with proper resampling of SpO2."""
        X_list = []
        y_list = []
        
        for w in data_windows:
            # Get signals
            nasal = w['nasal_flow']
            thoracic = w['thoracic']
            spo2 = w['spo2']
            
            # Resample SpO2 to match respiratory signals (32 Hz)
            if len(spo2) != len(nasal):
                # Upsample SpO2 using interpolation
                x_old = np.linspace(0, 1, len(spo2))
                x_new = np.linspace(0, 1, len(nasal))
                spo2 = np.interp(x_new, x_old, spo2)
            
            # Stack signals as channels
            signals = np.column_stack([nasal, thoracic, spo2])
            X_list.append(signals)
            y_list.append(w['label'])
        
        X = np.array(X_list)
        X = np.nan_to_num(X, nan=0.0)
        
        # Normalize
        n_samples, n_timesteps, n_channels = X.shape
        X_reshaped = X.reshape(-1, n_channels)
        
        if fit_scaler:
            X_reshaped = self.scaler.fit_transform(X_reshaped)
            y = self.label_encoder.fit_transform(y_list)
            self.class_names = self.label_encoder.classes_
        else:
            X_reshaped = self.scaler.transform(X_reshaped)
            y = self.label_encoder.transform(y_list)
        
        X = X_reshaped.reshape(n_samples, n_timesteps, n_channels)
        return X, y
    
    def plot_confusion_matrix(self, y_true, y_pred, title, save_path):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def leave_one_out_cv(self, data_windows, epochs=30, batch_size=16):
        """Perform Leave-One-Participant-Out Cross-Validation."""
        participants = sorted(set(w['participant'] for w in data_windows))
        print(f"\n{'='*60}")
        print(f"Leave-One-Participant-Out Cross-Validation")
        print(f"{'='*60}")
        print(f"Participants: {participants}")
        print(f"Total windows: {len(data_windows)}")
        
        all_results = {}
        all_true = []
        all_pred = []
        
        for test_idx, test_participant in enumerate(participants):
            print(f"\n{'='*50}")
            print(f"Fold {test_idx+1}/{len(participants)} - Testing on {test_participant}")
            print(f"{'='*50}")
            
            # Split data
            train_data = [w for w in data_windows if w['participant'] != test_participant]
            test_data = [w for w in data_windows if w['participant'] == test_participant]
            
            print(f"Train: {len(train_data)} samples")
            print(f"Test:  {len(test_data)} samples")
            
            # Prepare data
            X_train, y_train = self.prepare_data(train_data, fit_scaler=True)
            X_test, y_test = self.prepare_data(test_data, fit_scaler=False)
            
            # Print class distribution
            train_dist = pd.Series(self.label_encoder.inverse_transform(y_train)).value_counts()
            test_dist = pd.Series(self.label_encoder.inverse_transform(y_test)).value_counts()
            print(f"\nTrain distribution:\n{train_dist}")
            print(f"Test distribution:\n{test_dist}")
            
            # Class weights
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y_train), 
                y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])
            num_classes = len(np.unique(y_train))
            model = self.build_cnn_model(input_shape, num_classes)
            
            # Callbacks
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Store results
            all_results[test_participant] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            all_true.extend(y_test)
            all_pred.extend(y_pred)
            
            print(f"\nResults for {test_participant}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
        
        # Overall results
        print(f"\n{'='*60}")
        print("Overall Results")
        print(f"{'='*60}")
        
        overall_acc = accuracy_score(all_true, all_pred)
        overall_prec = precision_score(all_true, all_pred, average='weighted', zero_division=0)
        overall_rec = recall_score(all_true, all_pred, average='weighted', zero_division=0)
        overall_f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0)
        
        print(f"\nAverage Accuracy:  {overall_acc:.4f}")
        print(f"Average Precision: {overall_prec:.4f}")
        print(f"Average Recall:    {overall_rec:.4f}")
        print(f"Average F1-Score:  {overall_f1:.4f}")
        
        # Create summary dataframe
        results_df = pd.DataFrame([
            {'Participant': p, 'Accuracy': r['accuracy'], 
             'Precision': r['precision'], 'Recall': r['recall'], 'F1-Score': r['f1_score']}
            for p, r in all_results.items()
        ])
        
        return results_df, (all_true, all_pred)

def load_dataset(dataset_path):
    """Load dataset from pickle file."""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} windows")
    
    # Statistics
    participants = set(w['participant'] for w in data)
    labels = [w['label'] for w in data]
    
    print(f"\nDataset Statistics:")
    print(f"  Participants: {sorted(participants)}")
    print(f"  Label distribution:")
    for label, count in pd.Series(labels).value_counts().items():
        print(f"    {label}: {count}")
    
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True,
                       help='Path to dataset pickle file')
    parser.add_argument('-out_dir', type=str, default='models',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load data
    data_windows = load_dataset(args.dataset)
    
    # Train and evaluate
    classifier = BreathingClassifier()
    results_df, (y_true, y_pred) = classifier.leave_one_out_cv(
        data_windows, 
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save results
    print(f"\nSaving results to {args.out_dir}...")
    
    # Save per-participant results
    results_df.to_csv(os.path.join(args.out_dir, 'cv_results.csv'), index=False)
    print(f"  Saved CV results to {os.path.join(args.out_dir, 'cv_results.csv')}")
    
    # Plot overall confusion matrix
    classifier.plot_confusion_matrix(
        y_true, y_pred,
        title="Overall Confusion Matrix",
        save_path=os.path.join(args.out_dir, 'confusion_matrix.png')
    )
    print(f"  Saved confusion matrix to {os.path.join(args.out_dir, 'confusion_matrix.png')}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    print(f"\nAverage Accuracy:  {results_df['Accuracy'].mean():.4f} (+/- {results_df['Accuracy'].std():.4f})")
    print(f"Average Precision: {results_df['Precision'].mean():.4f} (+/- {results_df['Precision'].std():.4f})")
    print(f"Average Recall:    {results_df['Recall'].mean():.4f} (+/- {results_df['Recall'].std():.4f})")
    print(f"Average F1-Score:  {results_df['F1-Score'].mean():.4f} (+/- {results_df['F1-Score'].std():.4f})")

if __name__ == '__main__':
    main()