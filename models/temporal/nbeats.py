"""
N-BEATS - Neural Basis Expansion Analysis for Time Series.

N-BEATS is a deep learning model specifically designed for time series
forecasting. It uses fully connected networks with residual connections.
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


class NBEATSBlock(layers.Layer):
    """Single N-BEATS block."""
    
    def __init__(
        self,
        units: int,
        theta_units: int,
        backcast_length: int,
        forecast_length: int,
        share_weights: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.units = units
        self.theta_units = theta_units
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        
        # Fully connected stack
        self.fc_stack = [
            layers.Dense(units, activation='relu')
            for _ in range(4)
        ]
        
        # Theta layers
        self.theta_backcast = layers.Dense(theta_units, activation='linear')
        self.theta_forecast = layers.Dense(theta_units, activation='linear')
        
        # Basis expansion layers
        self.backcast_dense = layers.Dense(backcast_length, activation='linear')
        self.forecast_dense = layers.Dense(forecast_length, activation='linear')
    
    def call(self, inputs):
        x = inputs
        
        # FC stack
        for fc in self.fc_stack:
            x = fc(x)
        
        # Theta vectors
        theta_b = self.theta_backcast(x)
        theta_f = self.theta_forecast(x)
        
        # Basis expansion
        backcast = self.backcast_dense(theta_b)
        forecast = self.forecast_dense(theta_f)
        
        return backcast, forecast


class NBEATSModel(BaseModel):
    """
    N-BEATS model for time series forecasting.
    
    Architecture:
    - Multiple stacks of blocks
    - Each block produces backcast and forecast
    - Residual connections between blocks
    
    Usage:
        model = NBEATSModel(
            backcast_length=50,
            forecast_length=10,
            n_stacks=2,
            n_blocks=3
        )
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)
    """
    
    def __init__(
        self,
        name: str = "nbeats",
        backcast_length: int = 50,
        forecast_length: int = 10,
        n_stacks: int = 2,
        n_blocks: int = 3,
        units: int = 256,
        theta_units: int = 32,
        learning_rate: float = 0.001
    ):
        super().__init__(name)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for NBEATSModel")
        
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.n_stacks = n_stacks
        self.n_blocks = n_blocks
        self.units = units
        self.theta_units = theta_units
        self.learning_rate = learning_rate
        
        self.model: Optional[Model] = None
        self.scaler_mean: float = 0.0
        self.scaler_std: float = 1.0
        
        self._build_model()
    
    def _build_model(self):
        """Build N-BEATS architecture."""
        inputs = keras.Input(shape=(self.backcast_length,))
        
        # Initialize
        backcast = inputs
        forecast = tf.zeros((tf.shape(inputs)[0], self.forecast_length))
        
        # Stacks
        for stack_id in range(self.n_stacks):
            for block_id in range(self.n_blocks):
                block = NBEATSBlock(
                    units=self.units,
                    theta_units=self.theta_units,
                    backcast_length=self.backcast_length,
                    forecast_length=self.forecast_length,
                    name=f'stack{stack_id}_block{block_id}'
                )
                
                block_backcast, block_forecast = block(backcast)
                
                # Residual for backcast
                backcast = backcast - block_backcast
                # Accumulate forecast
                forecast = forecast + block_forecast
        
        self.model = Model(inputs=inputs, outputs=forecast)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        logger.info(f"Built N-BEATS model with {self.model.count_params()} parameters")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> 'NBEATSModel':
        """Train the model."""
        # Normalize
        self.scaler_mean = X.mean()
        self.scaler_std = X.std() + 1e-8
        
        X_norm = (X - self.scaler_mean) / self.scaler_std
        y_norm = (y - self.scaler_mean) / self.scaler_std
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        self.model.fit(
            X_norm, y_norm,
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
        """Generate forecast."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        X_norm = (X - self.scaler_mean) / self.scaler_std
        
        if len(X_norm.shape) == 1:
            X_norm = X_norm.reshape(1, -1)
        
        forecast_norm = self.model.predict(X_norm, verbose=0)
        forecast = forecast_norm * self.scaler_std + self.scaler_mean
        
        # Convert to trading signal
        current_price = X[-1] if len(X.shape) == 1 else X[0, -1]
        forecast_price = forecast[0, -1]
        
        expected_return = (forecast_price - current_price) / current_price
        
        direction = np.sign(expected_return)
        magnitude = abs(expected_return)
        confidence = min(1.0, magnitude * 10)  # Scale confidence
        
        return ModelPrediction(
            model_name=self.name,
            direction=float(direction),
            magnitude=float(magnitude),
            confidence=float(confidence),
            metadata={
                'forecast': forecast[0].tolist(),
                'expected_return': float(expected_return)
            }
        )
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length,
            'n_stacks': self.n_stacks,
            'n_blocks': self.n_blocks,
            'units': self.units,
            'theta_units': self.theta_units,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'weights': self.model.get_weights() if self.model else None
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.backcast_length = state['backcast_length']
        self.forecast_length = state['forecast_length']
        self.n_stacks = state['n_stacks']
        self.n_blocks = state['n_blocks']
        self.units = state['units']
        self.theta_units = state['theta_units']
        self.scaler_mean = state['scaler_mean']
        self.scaler_std = state['scaler_std']
        self._build_model()
        if state.get('weights'):
            self.model.set_weights(state['weights'])
