"""
Temporal Transformer - Transformer architecture for time series prediction.

Implements positional encoding and multi-head attention specifically
designed for financial time series.
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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel, ModelPrediction


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding for transformers."""
    
    def __init__(self, max_len: int = 5000, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
        # Compute positional encodings
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


class TransformerBlock(layers.Layer):
    """Single transformer encoder block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False, mask=None):
        # Self-attention
        attn_output = self.attention(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TemporalTransformer(BaseModel):
    """
    Transformer model for financial time series prediction.
    
    Architecture:
    - Input projection to d_model dimensions
    - Positional encoding
    - N transformer encoder blocks
    - Global pooling
    - Output heads for direction, magnitude, confidence
    
    Usage:
        model = TemporalTransformer(
            input_shape=(50, 10),
            d_model=64,
            num_heads=4,
            num_layers=3
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
    """
    
    def __init__(
        self,
        name: str = "temporal_transformer",
        input_shape: Tuple[int, int] = (50, 10),
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 128,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001
    ):
        super().__init__(name)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for TemporalTransformer")
        
        self.input_shape = input_shape
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model: Optional[Model] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        
        self._build_model()
    
    def _build_model(self):
        """Build the Transformer architecture."""
        inputs = keras.Input(shape=self.input_shape)
        
        # Project to d_model dimensions
        x = layers.Dense(self.d_model)(inputs)
        
        # Add positional encoding
        x = PositionalEncoding(max_len=self.input_shape[0], d_model=self.d_model)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate
            )(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        
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
        
        logger.info(f"Built Transformer model with {self.model.count_params()} parameters")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10
    ) -> 'TemporalTransformer':
        """Train the model."""
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
        
        return self
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Generate prediction from input features."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Normalize
        X_normalized = (X - self.scaler_mean) / self.scaler_std
        
        if len(X_normalized.shape) == 2:
            X_normalized = X_normalized.reshape(1, *X_normalized.shape)
        
        direction, magnitude, confidence = self.model.predict(X_normalized, verbose=0)
        
        return ModelPrediction(
            model_name=self.name,
            direction=float(direction[0, 0]),
            magnitude=float(magnitude[0, 0]),
            confidence=float(confidence[0, 0]),
            metadata={}
        )
    
    def _get_state(self) -> Dict[str, Any]:
        state = {
            'input_shape': self.input_shape,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
        }
        if self.model:
            state['weights'] = self.model.get_weights()
        return state
    
    def _set_state(self, state: Dict[str, Any]):
        self.input_shape = state['input_shape']
        self.d_model = state['d_model']
        self.num_heads = state['num_heads']
        self.num_layers = state['num_layers']
        self.ff_dim = state['ff_dim']
        self.dropout_rate = state['dropout_rate']
        self.learning_rate = state['learning_rate']
        self.scaler_mean = state.get('scaler_mean')
        self.scaler_std = state.get('scaler_std')
        self._build_model()
        if 'weights' in state:
            self.model.set_weights(state['weights'])
