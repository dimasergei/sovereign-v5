"""
Temporal Convolutional Network (TCN) - Dilated causal convolutions for sequences.

TCNs can capture long-range dependencies with fewer parameters than RNNs
and are faster to train due to parallelization.
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


class CausalConv1D(layers.Layer):
    """Causal 1D convolution with dilation."""
    
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        dilation_rate: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
        # Padding for causal convolution
        self.padding = (kernel_size - 1) * dilation_rate
        
        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal'
        )
    
    def call(self, inputs):
        return self.conv(inputs)


class ResidualBlock(layers.Layer):
    """TCN residual block with dilated causal convolutions."""
    
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.conv1 = CausalConv1D(filters, kernel_size, dilation_rate)
        self.conv2 = CausalConv1D(filters, kernel_size, dilation_rate)
        
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
        self.downsample = None
        self.filters = filters
    
    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.downsample = layers.Conv1D(self.filters, 1)
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = tf.nn.relu(x)
        x = self.dropout2(x, training=training)
        
        # Residual connection
        if self.downsample is not None:
            inputs = self.downsample(inputs)
        
        return tf.nn.relu(x + inputs)


class TCNModel(BaseModel):
    """
    Temporal Convolutional Network for time series prediction.
    
    Architecture:
    - Stack of residual blocks with increasing dilation
    - Dilation rates: 1, 2, 4, 8, 16, ... (exponential)
    - Each block has 2 causal dilated convolutions
    - Global pooling and dense output layers
    
    Usage:
        model = TCNModel(
            input_shape=(50, 10),
            num_filters=64,
            kernel_size=3,
            num_layers=6
        )
        model.fit(X_train, y_train)
    """
    
    def __init__(
        self,
        name: str = "tcn",
        input_shape: Tuple[int, int] = (50, 10),
        num_filters: int = 64,
        kernel_size: int = 3,
        num_layers: int = 6,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001
    ):
        super().__init__(name)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for TCNModel")
        
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model: Optional[Model] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        
        self._build_model()
    
    def _build_model(self):
        """Build TCN architecture."""
        inputs = keras.Input(shape=self.input_shape)
        
        x = inputs
        
        # Residual blocks with exponentially increasing dilation
        for i in range(self.num_layers):
            dilation_rate = 2 ** i
            x = ResidualBlock(
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_rate,
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
            }
        )
        
        # Calculate receptive field
        receptive_field = 1 + (self.kernel_size - 1) * sum(2**i for i in range(self.num_layers))
        logger.info(f"TCN receptive field: {receptive_field} timesteps")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10
    ) -> 'TCNModel':
        """Train the model."""
        self.scaler_mean = X.mean(axis=(0, 1))
        self.scaler_std = X.std(axis=(0, 1)) + 1e-8
        X_normalized = (X - self.scaler_mean) / self.scaler_std
        
        y_direction = y[:, 0:1]
        y_magnitude = y[:, 1:2]
        y_confidence = y[:, 2:3] if y.shape[1] > 2 else np.ones((len(y), 1))
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            )
        ]
        
        self.model.fit(
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
        return self
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
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
        return {
            'input_shape': self.input_shape,
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'weights': self.model.get_weights() if self.model else None
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.input_shape = state['input_shape']
        self.num_filters = state['num_filters']
        self.kernel_size = state['kernel_size']
        self.num_layers = state['num_layers']
        self.dropout_rate = state['dropout_rate']
        self.scaler_mean = state.get('scaler_mean')
        self.scaler_std = state.get('scaler_std')
        self._build_model()
        if state.get('weights'):
            self.model.set_weights(state['weights'])
