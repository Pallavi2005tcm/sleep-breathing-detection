"""
ConvLSTM Model for breathing irregularity classification.
Alternative architecture combining CNN and LSTM layers.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_convlstm_model(input_shape=(960, 3), num_classes=3):
    """
    Create a hybrid CNN-LSTM model.
    
    Args:
        input_shape: Tuple of (time_steps, n_channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Reshape for ConvLSTM (samples, time, height, width, channels)
        layers.Reshape((30, 32, 3, 1), input_shape=input_shape),
        
        # ConvLSTM layers
        layers.ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=False),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("ConvLSTM Model Architecture:")
    model = create_convlstm_model()
    model.summary()