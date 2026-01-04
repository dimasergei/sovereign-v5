#!/usr/bin/env python3
"""
Sovereign V5 ML Model Training Pipeline

Trains all ML models for signal generation:
- HMM Regime Detection
- LSTM with Attention
- Temporal Transformer
- PPO Reinforcement Learning
- Ensemble Meta-Learner

Usage:
    python scripts/train_models.py --all                    # Train all models
    python scripts/train_models.py --model hmm              # Train single model
    python scripts/train_models.py --symbols XAUUSD,EURUSD  # Specific symbols
    python scripts/train_models.py --years 2 --timeframe H1 # Custom data range
    python scripts/train_models.py --dry-run                # Validate setup only
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Optional, Tuple, Any
import logging
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('MLTrainer')

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    'symbols': ['XAUUSD', 'EURUSD', 'NAS100', 'BTCUSD'],
    'timeframe': 'H1',
    'years_of_data': 2,
    'train_test_split': 0.8,
    'validation_split': 0.1,

    # HMM Config
    'hmm': {
        'n_regimes': 4,
        'n_iter': 100,
        'covariance_type': 'full',
    },

    # LSTM Config
    'lstm': {
        'sequence_length': 50,
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.2,
        'early_stopping_patience': 10,
    },

    # Transformer Config
    'transformer': {
        'sequence_length': 50,
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.2,
        'early_stopping_patience': 10,
    },

    # PPO Config (simplified for initial training)
    'ppo': {
        'total_timesteps': 50000,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'clip_range': 0.2,
    },

    # Ensemble Config
    'ensemble': {
        'min_confidence': 0.5,
        'agreement_threshold': 0.6,
    }
}

MODEL_SAVE_PATH = PROJECT_ROOT / 'storage' / 'models'
CACHE_PATH = PROJECT_ROOT / 'data' / 'training_cache'

# ============================================================================
# IMPORTS - Check availability
# ============================================================================

def check_dependencies() -> Dict[str, bool]:
    """Check which ML libraries are available."""
    deps = {}

    try:
        from hmmlearn import hmm
        deps['hmm'] = True
    except ImportError:
        deps['hmm'] = False
        logger.warning("hmmlearn not installed - HMM training disabled")

    try:
        import tensorflow as tf
        deps['tensorflow'] = True
    except ImportError:
        deps['tensorflow'] = False
        logger.warning("TensorFlow not installed - LSTM/Transformer training disabled")

    try:
        import torch
        deps['torch'] = True
    except ImportError:
        deps['torch'] = False
        logger.warning("PyTorch not installed - PPO training disabled")

    return deps

AVAILABLE_DEPS = check_dependencies()

# ============================================================================
# DATA LOADING
# ============================================================================

def generate_synthetic_data(
    symbols: List[str],
    bars_per_symbol: int = 5000
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic OHLCV data for training when MT5 is not available.

    Uses geometric Brownian motion with regime switching for realistic price dynamics.
    """
    logger.info("Generating synthetic training data...")

    data = {}

    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)

        # Base parameters by symbol type
        if 'USD' in symbol and symbol.startswith(('XAU', 'XAG')):
            # Metals
            base_price = 2000 if 'XAU' in symbol else 25
            volatility = 0.015
        elif 'BTC' in symbol or 'ETH' in symbol:
            # Crypto
            base_price = 45000 if 'BTC' in symbol else 2500
            volatility = 0.025
        elif any(idx in symbol for idx in ['NAS', 'SPX', 'UK', 'US30']):
            # Indices
            base_price = 15000
            volatility = 0.012
        else:
            # Forex
            base_price = 1.10
            volatility = 0.008

        # Generate price series with regime switching
        prices = [base_price]
        regime = 0  # 0=trending up, 1=trending down, 2=ranging

        for i in range(bars_per_symbol - 1):
            # Regime switching
            if np.random.random() < 0.02:
                regime = np.random.randint(0, 3)

            # Drift based on regime
            if regime == 0:
                drift = 0.0002
            elif regime == 1:
                drift = -0.0002
            else:
                drift = 0

            # Random return
            ret = drift + volatility * np.random.randn()
            prices.append(prices[-1] * (1 + ret))

        prices = np.array(prices)

        # Generate OHLCV
        dates = pd.date_range(
            end=datetime.now(),
            periods=bars_per_symbol,
            freq='1h'
        )

        # Create realistic OHLC from close
        noise = volatility * 0.3
        df = pd.DataFrame({
            'time': dates,
            'open': prices * (1 + noise * np.random.randn(len(prices))),
            'high': prices * (1 + abs(noise * np.random.randn(len(prices)))),
            'low': prices * (1 - abs(noise * np.random.randn(len(prices)))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(prices)).astype(float)
        })

        # Ensure high >= close >= low
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        data[symbol] = df
        logger.info(f"  {symbol}: {len(df)} bars generated")

    return data


def fetch_mt5_data(
    symbols: List[str],
    timeframe: str = 'H1',
    bars_per_symbol: int = 5000
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Fetch real OHLCV data from MetaTrader 5.

    Uses GFT MT5 instance which requires .x suffix for symbols.

    Args:
        symbols: List of base symbols (e.g., ['XAUUSD', 'EURUSD'])
        timeframe: MT5 timeframe string (H1, M15, D1, etc.)
        bars_per_symbol: Number of bars to fetch per symbol

    Returns:
        Dict mapping symbol to DataFrame, or None if MT5 unavailable
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.warning("MetaTrader5 package not installed")
        return None

    # Initialize MT5
    if not mt5.initialize():
        logger.warning(f"MT5 initialization failed: {mt5.last_error()}")
        return None

    logger.info("Connected to MT5, fetching real market data...")

    # Map timeframe string to MT5 constant
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1,
    }

    mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

    data = {}

    for symbol in symbols:
        # Add .x suffix for GFT MT5 instance
        mt5_symbol = f"{symbol}.x"

        # Ensure symbol is selected
        if not mt5.symbol_select(mt5_symbol, True):
            logger.warning(f"  {mt5_symbol}: Failed to select symbol, skipping")
            continue

        # Fetch OHLCV data
        rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_timeframe, 0, bars_per_symbol)

        if rates is None or len(rates) == 0:
            logger.warning(f"  {mt5_symbol}: No data returned, skipping")
            continue

        # Convert to DataFrame
        df = pd.DataFrame(rates)

        # MT5 returns: time, open, high, low, close, tick_volume, spread, real_volume
        # Rename to match expected format
        df = df.rename(columns={
            'tick_volume': 'volume'
        })

        # Convert time from unix timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Keep only required columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

        # Store with base symbol name (without .x suffix)
        data[symbol] = df
        logger.info(f"  {symbol}: {len(df)} bars fetched from MT5")

    mt5.shutdown()

    if not data:
        logger.warning("No data fetched from MT5 for any symbol")
        return None

    return data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical features for model training."""
    df = df.copy()

    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_50'] = df['returns'].rolling(50).std()

    # Moving averages
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()

    # Price relative to MAs
    df['close_sma20_ratio'] = df['close'] / df['sma_20']
    df['close_sma50_ratio'] = df['close'] / df['sma_50']
    df['sma20_sma50_ratio'] = df['sma_20'] / df['sma_50']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # Momentum
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    # Volume features
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1)

    # Target: future return (next bar direction)
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1
    df['future_direction'] = np.sign(df['future_return'])

    # Drop NaN rows
    df = df.dropna()

    return df


def clean_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Remove NaN and infinity values from training data."""
    original_len = len(df)

    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN
    df = df.dropna()

    dropped = original_len - len(df)
    if dropped > 0:
        logger.info(f"    {symbol}: Dropped {dropped} rows with NaN/inf ({dropped/original_len*100:.1f}%)")

    return df


def prepare_sequences(
    df: pd.DataFrame,
    sequence_length: int = 50,
    feature_cols: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert DataFrame to sequences for LSTM/Transformer."""

    if feature_cols is None:
        # Default feature columns
        feature_cols = [
            'returns', 'log_returns', 'volatility_20', 'volatility_50',
            'close_sma20_ratio', 'close_sma50_ratio', 'sma20_sma50_ratio',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'atr_pct', 'bb_width', 'bb_position',
            'momentum_10', 'momentum_20', 'volume_ratio'
        ]

    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    features = df[feature_cols].values
    targets = df['future_direction'].values

    # Normalize features
    mean = np.nanmean(features, axis=0)
    std = np.nanstd(features, axis=0) + 1e-8
    features = (features - mean) / std

    # Replace any remaining NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(features) - 1):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])

    return np.array(X), np.array(y)


def train_test_split_temporal(
    data: Dict[str, pd.DataFrame],
    train_ratio: float = 0.8
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Split data temporally (no lookahead bias)."""

    train_data = {}
    test_data = {}

    for symbol, df in data.items():
        split_idx = int(len(df) * train_ratio)
        train_data[symbol] = df.iloc[:split_idx].copy()
        test_data[symbol] = df.iloc[split_idx:].copy()
        logger.info(f"  {symbol}: {len(train_data[symbol])} train, {len(test_data[symbol])} test")

    return train_data, test_data


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_hmm(data: Dict[str, pd.DataFrame], config: dict) -> Any:
    """Train HMM regime detection model."""

    if not AVAILABLE_DEPS.get('hmm', False):
        logger.error("hmmlearn not available, skipping HMM training")
        return None

    logger.info("=" * 60)
    logger.info("Training HMM Regime Model")
    logger.info("=" * 60)

    from models.regime.hmm import HMMRegimeModel

    model = HMMRegimeModel(
        n_regimes=config['n_regimes'],
        n_iter=config['n_iter'],
        covariance_type=config['covariance_type']
    )

    # Combine all symbols for regime training
    observations = []
    for symbol, df in data.items():
        if 'returns' not in df.columns:
            df = add_features(df)

        returns = df['returns'].dropna().values
        volatility = df['volatility_20'].dropna().values

        min_len = min(len(returns), len(volatility))
        if min_len > 100:
            obs = np.column_stack([returns[-min_len:], volatility[-min_len:]])
            observations.append(obs)
            logger.info(f"  {symbol}: {min_len} observations")

    if not observations:
        logger.error("No valid observations for HMM training")
        return None

    combined = np.vstack(observations)
    combined = np.nan_to_num(combined, nan=0.0)
    logger.info(f"Training on {len(combined)} total observations")

    try:
        model.fit(combined)

        # Save
        save_path = MODEL_SAVE_PATH / 'hmm' / 'hmm_regime.joblib'
        model.save(str(save_path))
        logger.info(f"Saved HMM model to {save_path}")

        return model
    except Exception as e:
        logger.error(f"HMM training failed: {e}")
        return None


def train_lstm(data: Dict[str, pd.DataFrame], config: dict) -> Any:
    """Train LSTM with attention model."""

    if not AVAILABLE_DEPS.get('tensorflow', False):
        logger.error("TensorFlow not available, skipping LSTM training")
        return None

    logger.info("=" * 60)
    logger.info("Training LSTM Attention Model")
    logger.info("=" * 60)

    from models.temporal.lstm_attention import LSTMAttentionModel

    # Prepare sequences from all symbols
    X_all, y_all = [], []
    for symbol, df in data.items():
        if 'returns' not in df.columns:
            df = add_features(df)

        X, y = prepare_sequences(df, config['sequence_length'])
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
            logger.info(f"  {symbol}: {len(X)} sequences")

    if not X_all:
        logger.error("No valid sequences for LSTM training")
        return None

    X_combined = np.vstack(X_all)
    y_combined = np.concatenate(y_all)

    # Shuffle
    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]

    logger.info(f"Total training samples: {len(X_combined)}")
    logger.info(f"Feature shape: {X_combined.shape}")

    # Initialize model
    input_shape = (X_combined.shape[1], X_combined.shape[2])

    try:
        model = LSTMAttentionModel(input_shape=input_shape)

        # Prepare targets (direction as classification: -1, 0, +1)
        # Convert to 3-class one-hot or regression
        y_train = y_combined.reshape(-1, 1)

        # Add dummy magnitude and confidence columns if needed
        y_train_full = np.hstack([
            y_train,  # direction
            np.abs(y_combined.reshape(-1, 1)) * 0.01,  # magnitude proxy
            np.ones((len(y_combined), 1)) * 0.5  # confidence proxy
        ])

        model.fit(
            X_combined, y_train_full,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=config['validation_split'],
            early_stopping_patience=config['early_stopping_patience']
        )

        # Save
        save_path = MODEL_SAVE_PATH / 'lstm' / 'lstm_attention.joblib'
        model.save(str(save_path))
        logger.info(f"Saved LSTM model to {save_path}")

        return model
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_transformer(data: Dict[str, pd.DataFrame], config: dict) -> Any:
    """Train Transformer model."""

    if not AVAILABLE_DEPS.get('tensorflow', False):
        logger.error("TensorFlow not available, skipping Transformer training")
        return None

    logger.info("=" * 60)
    logger.info("Training Temporal Transformer Model")
    logger.info("=" * 60)

    from models.temporal.transformer import TemporalTransformer

    # Same sequence preparation as LSTM
    X_all, y_all = [], []
    for symbol, df in data.items():
        if 'returns' not in df.columns:
            df = add_features(df)

        X, y = prepare_sequences(df, config['sequence_length'])
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
            logger.info(f"  {symbol}: {len(X)} sequences")

    if not X_all:
        logger.error("No valid sequences for Transformer training")
        return None

    X_combined = np.vstack(X_all)
    y_combined = np.concatenate(y_all)

    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]

    logger.info(f"Total training samples: {len(X_combined)}")

    input_shape = (X_combined.shape[1], X_combined.shape[2])

    try:
        model = TemporalTransformer(input_shape=input_shape)

        y_train_full = np.hstack([
            y_combined.reshape(-1, 1),
            np.abs(y_combined.reshape(-1, 1)) * 0.01,
            np.ones((len(y_combined), 1)) * 0.5
        ])

        model.fit(
            X_combined, y_train_full,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=config['validation_split'],
            early_stopping_patience=config['early_stopping_patience']
        )

        save_path = MODEL_SAVE_PATH / 'transformer' / 'transformer.joblib'
        model.save(str(save_path))
        logger.info(f"Saved Transformer model to {save_path}")

        return model
    except Exception as e:
        logger.error(f"Transformer training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_ppo(data: Dict[str, pd.DataFrame], config: dict) -> Any:
    """Train PPO reinforcement learning model."""

    if not AVAILABLE_DEPS.get('torch', False):
        logger.error("PyTorch not available, skipping PPO training")
        return None

    logger.info("=" * 60)
    logger.info("Training PPO Agent")
    logger.info("=" * 60)

    # PPO training is more complex - requires environment
    # For now, we'll skip this and use the other models
    logger.warning("PPO training requires custom environment setup")
    logger.warning("Skipping PPO for initial release - use HMM/LSTM/Transformer")

    return None


def create_ensemble(
    hmm_model: Any,
    lstm_model: Any,
    transformer_model: Any,
    config: dict
) -> Any:
    """Create and save ensemble meta-learner."""

    logger.info("=" * 60)
    logger.info("Creating Ensemble Meta-Learner")
    logger.info("=" * 60)

    from models.ensemble import EnsembleMetaLearner

    models = {}

    if hmm_model is not None:
        models['hmm'] = hmm_model
        logger.info("  Added HMM model to ensemble")

    if lstm_model is not None:
        models['lstm'] = lstm_model
        logger.info("  Added LSTM model to ensemble")

    if transformer_model is not None:
        models['transformer'] = transformer_model
        logger.info("  Added Transformer model to ensemble")

    if not models:
        logger.error("No models available for ensemble")
        return None

    ensemble = EnsembleMetaLearner(
        models=models,
        min_confidence=config['min_confidence'],
        agreement_threshold=config['agreement_threshold']
    )

    # Save ensemble config (models are saved separately)
    save_path = MODEL_SAVE_PATH / 'ensemble' / 'ensemble_config.joblib'
    ensemble_config = {
        'model_names': list(models.keys()),
        'min_confidence': config['min_confidence'],
        'agreement_threshold': config['agreement_threshold'],
        'created_at': datetime.now().isoformat()
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble_config, save_path)
    logger.info(f"Saved ensemble config to {save_path}")

    return ensemble


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model: Any,
    test_data: Dict[str, pd.DataFrame],
    model_name: str
) -> Dict[str, float]:
    """Evaluate a trained model on test data."""

    logger.info(f"\nEvaluating {model_name}...")

    results = {}

    for symbol, df in test_data.items():
        if 'returns' not in df.columns:
            df = add_features(df)

        try:
            if model_name == 'hmm':
                # HMM predicts regimes, not directions
                returns = df['returns'].dropna().values
                volatility = df['volatility_20'].dropna().values
                min_len = min(len(returns), len(volatility))
                obs = np.column_stack([returns[-min_len:], volatility[-min_len:]])

                pred = model.predict(obs)
                # Check if regime prediction is useful
                results[symbol] = {'regime_prediction': 'available'}

            else:
                # LSTM/Transformer predict directions
                X, y_true = prepare_sequences(df, 50)
                if len(X) == 0:
                    continue

                correct = 0
                total = 0

                for i in range(len(X)):
                    try:
                        pred = model.predict(X[i:i+1])
                        pred_dir = np.sign(pred.direction) if hasattr(pred, 'direction') else 0
                        if pred_dir == y_true[i]:
                            correct += 1
                        total += 1
                    except:
                        pass

                if total > 0:
                    accuracy = correct / total
                    results[symbol] = {'accuracy': accuracy}
                    logger.info(f"  {symbol}: {accuracy:.2%} accuracy ({correct}/{total})")

        except Exception as e:
            logger.warning(f"  {symbol}: Evaluation failed - {e}")

    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_training_pipeline(args):
    """Execute complete training pipeline."""

    logger.info("=" * 60)
    logger.info("SOVEREIGN V5 ML TRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 60)

    config = TRAINING_CONFIG.copy()

    # Override from args
    if args.symbols:
        config['symbols'] = [s.strip() for s in args.symbols.split(',')]
    if args.timeframe:
        config['timeframe'] = args.timeframe
    if args.years:
        config['years_of_data'] = args.years

    logger.info(f"\nConfiguration:")
    logger.info(f"  Symbols: {config['symbols']}")
    logger.info(f"  Timeframe: {config['timeframe']}")
    logger.info(f"  Years of data: {config['years_of_data']}")
    logger.info(f"  Model to train: {args.model}")

    # Step 1: Load/Generate Data
    logger.info("\n[1/5] Loading Training Data...")

    # Calculate bars needed based on years of data
    # Assuming H1 timeframe: ~24 bars/day * 365 days * years
    bars_needed = int(24 * 365 * config['years_of_data'])

    # Try to fetch real MT5 data first, fall back to synthetic
    data = fetch_mt5_data(
        symbols=config['symbols'],
        timeframe=config['timeframe'],
        bars_per_symbol=bars_needed
    )

    if data is None:
        logger.info("MT5 data unavailable, using synthetic data for training")
        data = generate_synthetic_data(config['symbols'], bars_needed)
    else:
        logger.info(f"Successfully loaded real MT5 data for {len(data)} symbols")

    if not data:
        logger.error("No data loaded. Exiting.")
        return

    # Step 2: Add features
    logger.info("\n[2/5] Engineering Features...")
    for symbol in data:
        data[symbol] = add_features(data[symbol])
        data[symbol] = clean_data(data[symbol], symbol)
        logger.info(f"  {symbol}: {data[symbol].shape[1]} features, {len(data[symbol])} rows")

    # Step 3: Split
    logger.info("\n[3/5] Splitting Train/Test...")
    train_data, test_data = train_test_split_temporal(data, config['train_test_split'])

    # Step 4: Train models
    hmm_model, lstm_model, transformer_model = None, None, None

    if args.model in ['all', 'hmm']:
        logger.info("\n[4a/5] Training HMM...")
        hmm_model = train_hmm(train_data, config['hmm'])

    if args.model in ['all', 'lstm']:
        logger.info("\n[4b/5] Training LSTM...")
        lstm_model = train_lstm(train_data, config['lstm'])

    if args.model in ['all', 'transformer']:
        logger.info("\n[4c/5] Training Transformer...")
        transformer_model = train_transformer(train_data, config['transformer'])

    if args.model in ['all', 'ppo']:
        logger.info("\n[4d/5] Training PPO...")
        train_ppo(train_data, config['ppo'])

    # Step 5: Create ensemble
    if args.model == 'all':
        logger.info("\n[5/5] Creating Ensemble...")
        ensemble = create_ensemble(
            hmm_model, lstm_model, transformer_model,
            config['ensemble']
        )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    # List saved models
    saved_models = list(MODEL_SAVE_PATH.rglob("*.joblib"))
    logger.info(f"\nSaved models ({len(saved_models)}):")
    for path in saved_models:
        size = path.stat().st_size / 1024
        logger.info(f"  {path.relative_to(MODEL_SAVE_PATH)}: {size:.1f} KB")

    # Evaluate on test set
    if args.model == 'all' and not args.dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("TEST SET EVALUATION")
        logger.info("=" * 60)

        if hmm_model:
            evaluate_model(hmm_model, test_data, 'hmm')
        if lstm_model:
            evaluate_model(lstm_model, test_data, 'lstm')
        if transformer_model:
            evaluate_model(transformer_model, test_data, 'transformer')


def validate_setup():
    """Validate training setup without actually training."""

    logger.info("=" * 60)
    logger.info("DRY RUN - Validating Setup")
    logger.info("=" * 60)

    # Check directories
    logger.info("\nDirectories:")
    for name in ['hmm', 'lstm', 'transformer', 'ppo', 'ensemble']:
        path = MODEL_SAVE_PATH / name
        exists = path.exists()
        logger.info(f"  {name}: {'✓' if exists else '✗'} {path}")

    # Check dependencies
    logger.info("\nDependencies:")
    for name, available in AVAILABLE_DEPS.items():
        logger.info(f"  {name}: {'✓' if available else '✗'}")

    # Check model imports
    logger.info("\nModel Imports:")

    try:
        from models.regime.hmm import HMMRegimeModel
        logger.info("  HMMRegimeModel: ✓")
    except Exception as e:
        logger.info(f"  HMMRegimeModel: ✗ ({e})")

    try:
        from models.temporal.lstm_attention import LSTMAttentionModel
        logger.info("  LSTMAttentionModel: ✓")
    except Exception as e:
        logger.info(f"  LSTMAttentionModel: ✗ ({e})")

    try:
        from models.temporal.transformer import TemporalTransformer
        logger.info("  TemporalTransformer: ✓")
    except Exception as e:
        logger.info(f"  TemporalTransformer: ✗ ({e})")

    try:
        from models.ensemble import EnsembleMetaLearner
        logger.info("  EnsembleMetaLearner: ✓")
    except Exception as e:
        logger.info(f"  EnsembleMetaLearner: ✗ ({e})")

    # Generate small test data
    logger.info("\nTest Data Generation:")
    try:
        data = generate_synthetic_data(['XAUUSD'], 100)
        data['XAUUSD'] = add_features(data['XAUUSD'])
        logger.info(f"  Synthetic data: ✓ ({data['XAUUSD'].shape})")
    except Exception as e:
        logger.info(f"  Synthetic data: ✗ ({e})")

    logger.info("\n✓ Setup validation complete")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Sovereign V5 ML Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/train_models.py --all                     # Train all models
    python scripts/train_models.py --model hmm               # Train HMM only
    python scripts/train_models.py --model lstm --epochs 100 # Train LSTM with custom epochs
    python scripts/train_models.py --dry-run                 # Validate setup
    python scripts/train_models.py --symbols XAUUSD,BTCUSD   # Custom symbols
        """
    )

    parser.add_argument(
        '--model', type=str, default='all',
        choices=['all', 'hmm', 'lstm', 'transformer', 'ppo', 'ensemble'],
        help='Which model to train (default: all)'
    )
    parser.add_argument(
        '--symbols', type=str, default=None,
        help='Comma-separated symbols (e.g., XAUUSD,EURUSD)'
    )
    parser.add_argument(
        '--timeframe', type=str, default=None,
        help='Timeframe (e.g., H1, H4, D1)'
    )
    parser.add_argument(
        '--years', type=int, default=None,
        help='Years of historical data'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate setup without training'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override epochs for deep learning models'
    )

    args = parser.parse_args()

    # Override epochs if specified
    if args.epochs:
        TRAINING_CONFIG['lstm']['epochs'] = args.epochs
        TRAINING_CONFIG['transformer']['epochs'] = args.epochs

    if args.dry_run:
        validate_setup()
    else:
        run_training_pipeline(args)
