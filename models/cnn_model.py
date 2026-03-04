"""
1D CNN Model for breathing irregularity classification.
This is the main model architecture used in train_model.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cnn_model(input_shape=(960, 3), num_classes=3):
    """
    Create a 1D CNN model for breathing pattern classification.
    
    Args:
        input_shape: Tuple of (time_steps, n_channels) - (960, 3) for 30s at 32Hz
        num_classes: Number of output classes (Normal, Hypopnea, Obstructive Apnea)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv1D(32, 3, activation='relu', input_shape=input_shape, padding='same'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        # Second convolutional block
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        # Third convolutional block
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        
        # Dense layers
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

def model_summary():
    """Print model summary."""
    model = create_cnn_model()
    model.summary()
    return model

if __name__ == "__main__":
    print("CNN Model Architecture:")
    model_summary()