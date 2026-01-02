"""
Black-Litterman Portfolio Model - Combining market equilibrium with views.

The Black-Litterman model provides a theoretically sound way to combine
quantitative views with market equilibrium returns.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy import linalg

logger = logging.getLogger(__name__)


@dataclass
class View:
    """Single view on asset returns."""
    assets: List[str]  # Assets involved
    weights: List[float]  # View weights (sum to 0 for relative)
    expected_return: float  # Expected return from the view
    confidence: float  # Confidence level (0-1)
    
    def is_absolute(self) -> bool:
        """Check if view is absolute (single asset)."""
        return len(self.assets) == 1
    
    def is_relative(self) -> bool:
        """Check if view is relative (comparing assets)."""
        return len(self.assets) > 1 and abs(sum(self.weights)) < 0.001


class BlackLittermanModel:
    """
    Black-Litterman portfolio allocation model.
    
    Combines market equilibrium (CAPM) returns with investor views
    to generate posterior expected returns for portfolio optimization.
    
    Key features:
    - Market-implied equilibrium returns
    - View incorporation with confidence levels
    - Uncertainty-adjusted returns
    
    Usage:
        bl = BlackLittermanModel(risk_free_rate=0.02)
        
        # Fit to market data
        bl.fit(returns_df, market_caps)
        
        # Add views
        bl.add_view(['BTC'], [1.0], 0.10, confidence=0.8)  # BTC returns 10%
        bl.add_view(['ETH', 'BTC'], [1.0, -1.0], 0.05, confidence=0.6)  # ETH beats BTC by 5%
        
        # Get posterior returns and optimal weights
        posterior_returns = bl.get_posterior_returns()
        weights = bl.get_optimal_weights()
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        risk_aversion: float = 2.5,
        tau: float = 0.05
    ):
        """
        Initialize Black-Litterman model.
        
        Args:
            risk_free_rate: Annual risk-free rate
            risk_aversion: Market risk aversion coefficient
            tau: Scaling factor for prior uncertainty (typically 0.01-0.1)
        """
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.tau = tau
        
        # Fitted parameters
        self.assets: List[str] = []
        self.covariance: Optional[np.ndarray] = None
        self.equilibrium_returns: Optional[np.ndarray] = None
        self.market_weights: Optional[np.ndarray] = None
        
        # Views
        self.views: List[View] = []
        
        # Posterior
        self.posterior_returns: Optional[np.ndarray] = None
        self.posterior_covariance: Optional[np.ndarray] = None
    
    def fit(
        self,
        returns: pd.DataFrame,
        market_caps: Optional[Dict[str, float]] = None,
        annualization_factor: int = 252
    ) -> 'BlackLittermanModel':
        """
        Fit model to historical returns.
        
        Args:
            returns: DataFrame of asset returns
            market_caps: Dict of asset -> market cap (for market weights)
            annualization_factor: Factor to annualize returns/covariance
        """
        self.assets = returns.columns.tolist()
        n_assets = len(self.assets)
        
        # Annualized covariance
        self.covariance = returns.cov().values * annualization_factor
        
        # Market weights
        if market_caps:
            total_cap = sum(market_caps.get(a, 1.0) for a in self.assets)
            self.market_weights = np.array([
                market_caps.get(a, 1.0) / total_cap for a in self.assets
            ])
        else:
            # Equal weights if no market caps
            self.market_weights = np.ones(n_assets) / n_assets
        
        # Reverse optimization to get equilibrium returns
        # Pi = delta * Sigma * w_mkt
        self.equilibrium_returns = self.risk_aversion * self.covariance @ self.market_weights
        
        logger.info(f"Black-Litterman fitted with {n_assets} assets")
        
        return self
    
    def add_view(
        self,
        assets: List[str],
        weights: List[float],
        expected_return: float,
        confidence: float = 0.5
    ):
        """
        Add a view on asset returns.
        
        Args:
            assets: List of assets in the view
            weights: View weights (positive for long, negative for short)
            expected_return: Expected return from the view
            confidence: Confidence in the view (0-1)
        """
        view = View(
            assets=assets,
            weights=weights,
            expected_return=expected_return,
            confidence=confidence
        )
        self.views.append(view)
        
        logger.info(f"Added view: {assets} with return {expected_return:.2%}")
    
    def clear_views(self):
        """Clear all views."""
        self.views.clear()
    
    def get_posterior_returns(self) -> np.ndarray:
        """
        Calculate posterior expected returns incorporating views.
        
        Returns:
            Array of posterior expected returns
        """
        if self.covariance is None:
            raise ValueError("Model not fitted")
        
        n_assets = len(self.assets)
        
        if not self.views:
            # No views, return equilibrium
            self.posterior_returns = self.equilibrium_returns
            self.posterior_covariance = self.covariance
            return self.equilibrium_returns
        
        # Build view matrices
        n_views = len(self.views)
        P = np.zeros((n_views, n_assets))  # Pick matrix
        Q = np.zeros(n_views)  # View returns
        omega_diag = np.zeros(n_views)  # View uncertainty
        
        for i, view in enumerate(self.views):
            for asset, weight in zip(view.assets, view.weights):
                if asset in self.assets:
                    j = self.assets.index(asset)
                    P[i, j] = weight
            
            Q[i] = view.expected_return
            
            # Omega (view uncertainty) from confidence
            # Higher confidence = lower uncertainty
            view_variance = P[i] @ self.covariance @ P[i].T
            omega_diag[i] = view_variance * (1 - view.confidence) / view.confidence
        
        Omega = np.diag(omega_diag)
        
        # Black-Litterman formula
        # E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
        
        tau_sigma = self.tau * self.covariance
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)
        
        # Posterior precision
        posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P
        posterior_covariance = np.linalg.inv(posterior_precision)
        
        # Posterior mean
        posterior_returns = posterior_covariance @ (
            tau_sigma_inv @ self.equilibrium_returns + P.T @ omega_inv @ Q
        )
        
        self.posterior_returns = posterior_returns
        self.posterior_covariance = posterior_covariance + self.covariance
        
        return posterior_returns
    
    def get_optimal_weights(
        self,
        target_return: float = None,
        risk_aversion: float = None
    ) -> np.ndarray:
        """
        Get optimal portfolio weights using posterior returns.
        
        Args:
            target_return: Target portfolio return (if None, maximize Sharpe)
            risk_aversion: Override risk aversion coefficient
            
        Returns:
            Array of optimal portfolio weights
        """
        if self.posterior_returns is None:
            self.get_posterior_returns()
        
        delta = risk_aversion or self.risk_aversion
        
        # Mean-variance optimization: w* = (1/delta) * Sigma^-1 * mu
        cov_inv = np.linalg.inv(self.posterior_covariance)
        raw_weights = (1 / delta) * cov_inv @ self.posterior_returns
        
        # Normalize to sum to 1
        weights = raw_weights / raw_weights.sum()
        
        # Apply constraints (long-only, max position)
        weights = np.clip(weights, 0, 0.4)  # Max 40% per asset
        weights = weights / weights.sum()
        
        return weights
    
    def get_allocation_dict(self) -> Dict[str, float]:
        """Get allocation as dictionary."""
        weights = self.get_optimal_weights()
        return {asset: float(w) for asset, w in zip(self.assets, weights)}
    
    def get_portfolio_stats(self, weights: np.ndarray = None) -> Dict[str, float]:
        """
        Get portfolio statistics for given weights.
        
        Returns:
            Dict with expected return, volatility, Sharpe ratio
        """
        if weights is None:
            weights = self.get_optimal_weights()
        
        if self.posterior_returns is None:
            self.get_posterior_returns()
        
        expected_return = weights @ self.posterior_returns
        volatility = np.sqrt(weights @ self.posterior_covariance @ weights)
        sharpe = (expected_return - self.risk_free_rate) / volatility
        
        return {
            'expected_return': float(expected_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe)
        }


class ViewGenerator:
    """
    Generate views from quantitative signals.
    
    Converts model predictions to Black-Litterman views.
    """
    
    @staticmethod
    def from_predictions(
        predictions: Dict[str, float],
        confidence_map: Dict[str, float] = None,
        scale: float = 0.1
    ) -> List[View]:
        """
        Generate absolute views from model predictions.
        
        Args:
            predictions: Dict of asset -> predicted return
            confidence_map: Dict of asset -> confidence
            scale: Scaling factor for predictions
            
        Returns:
            List of View objects
        """
        views = []
        
        for asset, prediction in predictions.items():
            confidence = confidence_map.get(asset, 0.5) if confidence_map else 0.5
            
            view = View(
                assets=[asset],
                weights=[1.0],
                expected_return=prediction * scale,
                confidence=confidence
            )
            views.append(view)
        
        return views
    
    @staticmethod
    def from_rankings(
        rankings: List[str],
        return_spread: float = 0.05,
        confidence: float = 0.6
    ) -> List[View]:
        """
        Generate relative views from asset rankings.
        
        Args:
            rankings: List of assets from best to worst
            return_spread: Expected return difference between adjacent assets
            confidence: Confidence in rankings
            
        Returns:
            List of relative View objects
        """
        views = []
        
        for i in range(len(rankings) - 1):
            view = View(
                assets=[rankings[i], rankings[i + 1]],
                weights=[1.0, -1.0],
                expected_return=return_spread,
                confidence=confidence
            )
            views.append(view)
        
        return views
