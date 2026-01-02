"""
LSTM with Attention - Deep learning model for temporal pattern recognition.

Implements multi-head self-attention on top of LSTM layers for
capturing long-range dependencies in price series.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel, ModelPrediction


class AttentionLayer(layers.Layer):
    """Multi-head self-attention layer."""
    
    def __init__(self, units: int, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.head_dim = units // num_heads
        
        self.query_dense = layers.Dense(units)
        self.key_dense = layers.Dense(units)
        self.value_dense = layers.Dense(units)
        self.output_dense = layers.Dense(units)
    
    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Linear projections
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Reshape to (batch, heads, seq, head_dim)
        query = tf.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        
        key = tf.reshape(key, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        
        value = tf.reshape(value, (batch_size, seq_len, self.num_heads, self.head_dim))
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        
        # Attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        if mask is not None:
            scores += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        context = tf.matmul(attention_weights, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, seq_len, self.units))
        
        output = self.output_dense(context)
        return output, attention_weights


class LSTMAttentionModel(BaseModel):
    """
    LSTM with Multi-Head Attention for time series prediction.
    
    Architecture:
    - Input normalization
    - Bidirectional LSTM layers
    - Multi-head self-attention
    - Dense layers with dropout
    - Output: direction, magnitude, confidence
    
    Usage:
        model = LSTMAttentionModel(
            input_shape=(50, 10),  # 50 timesteps, 10 features
            lstm_units=[64, 32],
            attention_heads=4
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
    """
    
    def __init__(
        self,
        name: str = "lstm_attention",
        input_shape: Tuple[int, int] = (50, 10),
        lstm_units: List[int] = [64, 32],
        attention_units: int = 32,
        attention_heads: int = 4,
        dense_units: List[int] = [32, 16],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001
    ):
        super().__init__(name)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for LSTMAttentionModel")
        
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.attention_heads = attention_heads
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model: Optional[Model] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        
        self._build_model()
    
    def _build_model(self):
        """Build the LSTM-Attention architecture."""
        inputs = keras.Input(shape=self.input_shape)
        
        x = inputs
        
        # Bidirectional LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1) or True  # Need sequences for attention
            x = layers.Bidirectional(
                layers.LSTM(units, return_sequences=return_sequences, dropout=self.dropout_rate)
            )(x)
        
        # Multi-head attention
        attention_layer = AttentionLayer(self.attention_units, self.attention_heads)
        x, attention_weights = attention_layer(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        for units in self.dense_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output heads
        direction = layers.Dense(1, activation='tanh', name='direction')(x)
        magnitude = layers.Dense(1, activation='relu', name='magnitude')(x)
        confidence = layers.Dense(1, activation='sigmoid', name='confidence')(x)
        
        self.model = Model(inputs=inputs, outputs=[direction, magnitude, confidence])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                'direction': 'mse',
                'magnitude': 'mse',
                'confidence': 'binary_crossentropy'
            },
            loss_weights={'direction': 1.0, 'magnitude': 0.5, 'confidence': 0.5}
        )
        
        logger.info(f"Built LSTM-Attention model with {self.model.count_params()} parameters")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10
    ) -> 'LSTMAttentionModel':
        """
        Train the model.
        
        Args:
            X: Input features (samples, timesteps, features)
            y: Target values (samples, 3) - [direction, magnitude, was_profitable]
        """
        # Normalize inputs
        self.scaler_mean = X.mean(axis=(0, 1))
        self.scaler_std = X.std(axis=(0, 1)) + 1e-8
        X_normalized = (X - self.scaler_mean) / self.scaler_std
        
        # Prepare targets
        y_direction = y[:, 0:1]
        y_magnitude = y[:, 1:2]
        y_confidence = y[:, 2:3] if y.shape[1] > 2 else np.ones((len(y), 1))
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train
        history = self.model.fit(
            X_normalized,
            {'direction': y_direction, 'magnitude': y_magnitude, 'confidence': y_confidence},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        self.training_date = datetime.now()
        self.metadata['epochs_trained'] = len(history.history['loss'])
        self.metadata['final_loss'] = float(history.history['loss'][-1])
        
        logger.info(f"Training complete. Final loss: {self.metadata['final_loss']:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Generate prediction from input features."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Normalize
        X_normalized = (X - self.scaler_mean) / self.scaler_std
        
        # Ensure 3D shape
        if len(X_normalized.shape) == 2:
            X_normalized = X_normalized.reshape(1, *X_normalized.shape)
        
        # Predict
        direction, magnitude, confidence = self.model.predict(X_normalized, verbose=0)
        
        return ModelPrediction(
            model_name=self.name,
            direction=float(direction[0, 0]),
            magnitude=float(magnitude[0, 0]),
            confidence=float(confidence[0, 0]),
            metadata={
                'raw_direction': float(direction[0, 0]),
                'raw_magnitude': float(magnitude[0, 0]),
                'raw_confidence': float(confidence[0, 0])
            }
        )
    
    def _get_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        state = {
            'input_shape': self.input_shape,
            'lstm_units': self.lstm_units,
            'attention_units': self.attention_units,
            'attention_heads': self.attention_heads,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
        }
        
        if self.model is not None:
            state['weights'] = self.model.get_weights()
        
        return state
    
    def _set_state(self, state: Dict[str, Any]):
        """Set model state from serialization."""
        self.input_shape = state['input_shape']
        self.lstm_units = state['lstm_units']
        self.attention_units = state['attention_units']
        self.attention_heads = state['attention_heads']
        self.dense_units = state['dense_units']
        self.dropout_rate = state['dropout_rate']
        self.learning_rate = state['learning_rate']
        self.scaler_mean = state.get('scaler_mean')
        self.scaler_std = state.get('scaler_std')
        
        self._build_model()
        
        if 'weights' in state:
            self.model.set_weights(state['weights'])


def create_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    sequence_length: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        data: Feature array (samples, features)
        targets: Target array (samples, target_dims)
        sequence_length: Number of timesteps per sequence
        
    Returns:
        X: Sequences (num_sequences, sequence_length, features)
        y: Targets aligned with sequences
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(targets[i + sequence_length])
    
    return np.array(X), np.array(y)
