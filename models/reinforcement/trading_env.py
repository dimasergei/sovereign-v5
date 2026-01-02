"""
Trading Environment - Gym-compatible environment for RL agents.

Provides a standardized interface for training RL agents on trading tasks.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        logger.warning("Neither gymnasium nor gym installed")


@dataclass
class TradingConfig:
    """Trading environment configuration."""
    initial_balance: float = 10000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    max_position: float = 1.0  # Max position as fraction of balance
    leverage: float = 1.0
    
    # Risk limits
    max_drawdown: float = 0.08  # 8% max drawdown (GFT)
    daily_loss_limit: float = 0.05  # 5% daily loss (The5ers)
    
    # Reward shaping
    reward_scaling: float = 100.0
    risk_penalty: float = 0.1
    holding_cost: float = 0.0001  # Overnight cost


class TradingEnvironment:
    """
    Trading environment for reinforcement learning.
    
    State: [price_features..., position, balance, drawdown, ...]
    Action: Continuous [-1, 1] for position size (short to long)
    Reward: Risk-adjusted PnL with penalties
    
    Usage:
        env = TradingEnvironment(df, config)
        
        state = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: TradingConfig = None,
        feature_columns: List[str] = None,
        window_size: int = 50
    ):
        """
        Initialize trading environment.
        
        Args:
            df: OHLCV DataFrame with features
            config: TradingConfig instance
            feature_columns: Columns to use as state features
            window_size: Lookback window for state
        """
        self.df = df.copy()
        self.config = config or TradingConfig()
        self.window_size = window_size
        
        # Determine feature columns
        if feature_columns:
            self.feature_columns = feature_columns
        else:
            # Use all numeric columns except OHLCV
            exclude = ['open', 'high', 'low', 'close', 'volume']
            self.feature_columns = [
                c for c in df.columns 
                if c not in exclude and df[c].dtype in [np.float64, np.int64]
            ]
            if not self.feature_columns:
                # Fall back to returns-based features
                self.df['returns'] = self.df['close'].pct_change()
                self.df['volatility'] = self.df['returns'].rolling(20).std()
                self.feature_columns = ['returns', 'volatility']
        
        self.df = self.df.dropna()
        
        # State and action dimensions
        self.n_features = len(self.feature_columns)
        self.state_dim = self.n_features * self.window_size + 4  # +4 for account state
        self.action_dim = 1  # Position size
        
        # Define spaces if gym available
        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.state_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, 
                shape=(self.action_dim,), dtype=np.float32
            )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.window_size
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.high_water_mark = self.config.initial_balance
        self.daily_start_balance = self.config.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action and return next state, reward, done, info.
        
        Args:
            action: Position size [-1, 1]
        """
        action = np.clip(action, -1, 1)
        target_position = float(action[0]) if hasattr(action, '__len__') else float(action)
        
        # Current price
        current_price = self.df['close'].iloc[self.current_step]
        
        # Calculate position change
        position_change = target_position - self.position
        
        # Transaction costs
        transaction_cost = abs(position_change) * self.balance * (
            self.config.commission_rate + self.config.slippage_rate
        )
        
        # Move to next step
        self.current_step += 1
        
        if self.current_step >= len(self.df):
            done = True
            next_price = current_price
        else:
            done = False
            next_price = self.df['close'].iloc[self.current_step]
        
        # Calculate PnL
        price_return = (next_price - current_price) / current_price
        position_pnl = self.position * self.balance * price_return * self.config.leverage
        
        # Update balance
        self.balance += position_pnl - transaction_cost
        
        # Update position
        old_position = self.position
        self.position = target_position
        
        # Track trades
        if abs(position_change) > 0.01:
            self.total_trades += 1
            if position_pnl > 0:
                self.winning_trades += 1
        
        self.total_pnl += position_pnl - transaction_cost
        
        # Update high water mark
        self.high_water_mark = max(self.high_water_mark, self.balance)
        
        # Calculate drawdown
        drawdown = (self.high_water_mark - self.balance) / self.high_water_mark
        
        # Check risk limits
        if drawdown > self.config.max_drawdown:
            done = True
            logger.warning(f"Max drawdown exceeded: {drawdown:.2%}")
        
        daily_loss = (self.daily_start_balance - self.balance) / self.daily_start_balance
        if daily_loss > self.config.daily_loss_limit:
            done = True
            logger.warning(f"Daily loss limit exceeded: {daily_loss:.2%}")
        
        # Calculate reward
        reward = self._calculate_reward(
            pnl=position_pnl - transaction_cost,
            drawdown=drawdown,
            position_change=abs(position_change)
        )
        
        # Get next state
        next_state = self._get_state()
        
        # Info dict
        info = {
            'balance': self.balance,
            'position': self.position,
            'pnl': position_pnl - transaction_cost,
            'drawdown': drawdown,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_return': (self.balance - self.config.initial_balance) / self.config.initial_balance
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        # Feature window
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        features = self.df[self.feature_columns].iloc[start_idx:end_idx].values.flatten()
        
        # Normalize features
        features = np.nan_to_num(features, 0)
        
        # Account state
        normalized_balance = (self.balance - self.config.initial_balance) / self.config.initial_balance
        drawdown = (self.high_water_mark - self.balance) / self.high_water_mark
        
        account_state = np.array([
            self.position,
            normalized_balance,
            drawdown,
            self.total_trades / 100.0  # Normalized trade count
        ])
        
        state = np.concatenate([features, account_state])
        
        return state.astype(np.float32)
    
    def _calculate_reward(
        self,
        pnl: float,
        drawdown: float,
        position_change: float
    ) -> float:
        """
        Calculate reward with shaping.
        
        Components:
        - PnL (main signal)
        - Drawdown penalty
        - Trading cost penalty
        """
        # Normalize PnL
        pnl_reward = pnl / self.config.initial_balance * self.config.reward_scaling
        
        # Drawdown penalty
        drawdown_penalty = -drawdown * self.config.risk_penalty * self.config.reward_scaling
        
        # Trading frequency penalty (discourage overtrading)
        trade_penalty = -position_change * 0.01
        
        reward = pnl_reward + drawdown_penalty + trade_penalty
        
        return float(reward)
    
    def render(self, mode: str = 'human'):
        """Render environment state."""
        print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, "
              f"Position: {self.position:.2f}, "
              f"Return: {(self.balance/self.config.initial_balance - 1)*100:.2f}%")


class MultiAssetEnvironment:
    """
    Multi-asset trading environment.
    
    Handles portfolio of assets with correlation management.
    """
    
    def __init__(
        self,
        dfs: Dict[str, pd.DataFrame],
        config: TradingConfig = None,
        window_size: int = 50
    ):
        """
        Initialize multi-asset environment.
        
        Args:
            dfs: Dict of symbol -> DataFrame
            config: TradingConfig instance
            window_size: Lookback window
        """
        self.symbols = list(dfs.keys())
        self.n_assets = len(self.symbols)
        self.dfs = dfs
        self.config = config or TradingConfig()
        self.window_size = window_size
        
        # Align dataframes
        common_index = None
        for df in dfs.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        self.aligned_dfs = {
            symbol: df.loc[common_index] 
            for symbol, df in dfs.items()
        }
        
        self.n_steps = len(common_index)
        
        # State and action dimensions
        self.state_dim = self.n_assets * (window_size + 1) + 3  # +1 for position, +3 for account
        self.action_dim = self.n_assets
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.current_step = self.window_size
        self.balance = self.config.initial_balance
        self.positions = np.zeros(self.n_assets)
        self.high_water_mark = self.config.initial_balance
        
        return self._get_state()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action across all assets."""
        actions = np.clip(actions, -1, 1)
        
        # Normalize to not exceed max position
        total_exposure = np.abs(actions).sum()
        if total_exposure > 1:
            actions = actions / total_exposure
        
        # Get current prices
        current_prices = np.array([
            self.aligned_dfs[s]['close'].iloc[self.current_step]
            for s in self.symbols
        ])
        
        # Calculate transaction costs
        position_changes = np.abs(actions - self.positions)
        transaction_costs = position_changes.sum() * self.balance * (
            self.config.commission_rate + self.config.slippage_rate
        )
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # Get next prices
        next_prices = np.array([
            self.aligned_dfs[s]['close'].iloc[self.current_step]
            for s in self.symbols
        ])
        
        # Calculate returns
        returns = (next_prices - current_prices) / current_prices
        
        # Calculate PnL
        position_pnls = self.positions * self.balance * returns
        total_pnl = position_pnls.sum() - transaction_costs
        
        # Update balance
        self.balance += total_pnl
        
        # Update positions
        self.positions = actions.copy()
        
        # Update high water mark and calculate drawdown
        self.high_water_mark = max(self.high_water_mark, self.balance)
        drawdown = (self.high_water_mark - self.balance) / self.high_water_mark
        
        # Check limits
        if drawdown > self.config.max_drawdown:
            done = True
        
        # Calculate reward
        reward = total_pnl / self.config.initial_balance * self.config.reward_scaling
        reward -= drawdown * self.config.risk_penalty
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'balance': self.balance,
            'positions': self.positions.copy(),
            'pnl': total_pnl,
            'drawdown': drawdown,
            'returns': returns
        }
        
        return next_state, float(reward), done, info
    
    def _get_state(self) -> np.ndarray:
        """Get state observation."""
        features = []
        
        for symbol in self.symbols:
            df = self.aligned_dfs[symbol]
            returns = df['close'].pct_change().iloc[
                self.current_step - self.window_size:self.current_step
            ].values
            features.extend(returns)
        
        # Add positions
        features.extend(self.positions)
        
        # Add account state
        features.extend([
            (self.balance - self.config.initial_balance) / self.config.initial_balance,
            (self.high_water_mark - self.balance) / self.high_water_mark,
            self.current_step / self.n_steps
        ])
        
        return np.array(features, dtype=np.float32)
