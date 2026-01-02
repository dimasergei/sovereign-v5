"""
Hierarchical Risk Parity (HRP) - Machine learning-based portfolio allocation.

HRP uses hierarchical clustering to build a portfolio that's more stable
than traditional mean-variance optimization.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity portfolio allocation.
    
    Key features:
    - Hierarchical clustering of assets
    - Recursive bisection for weight allocation
    - More robust than mean-variance optimization
    - No need for return estimation
    
    Based on Lopez de Prado (2016).
    
    Usage:
        hrp = HierarchicalRiskParity()
        hrp.fit(returns_df)
        
        weights = hrp.get_weights()
        allocation = hrp.get_allocation_dict()
    """
    
    def __init__(
        self,
        linkage_method: str = 'ward',
        risk_measure: str = 'variance'
    ):
        """
        Initialize HRP model.
        
        Args:
            linkage_method: Clustering method ('ward', 'single', 'complete', 'average')
            risk_measure: Risk measure for allocation ('variance', 'std', 'cvar')
        """
        self.linkage_method = linkage_method
        self.risk_measure = risk_measure
        
        # Fitted parameters
        self.assets: List[str] = []
        self.returns: Optional[pd.DataFrame] = None
        self.covariance: Optional[np.ndarray] = None
        self.correlation: Optional[np.ndarray] = None
        self.linkage_matrix: Optional[np.ndarray] = None
        self.sorted_indices: Optional[List[int]] = None
        self.weights: Optional[np.ndarray] = None
    
    def fit(self, returns: pd.DataFrame) -> 'HierarchicalRiskParity':
        """
        Fit HRP model to returns data.
        
        Args:
            returns: DataFrame of asset returns
        """
        self.assets = returns.columns.tolist()
        self.returns = returns
        
        # Calculate covariance and correlation
        self.covariance = returns.cov().values
        self.correlation = returns.corr().values
        
        # Tree clustering
        self.linkage_matrix = self._get_linkage()
        
        # Quasi-diagonalization
        self.sorted_indices = self._get_quasi_diag()
        
        # Recursive bisection
        self.weights = self._get_recursive_bisection()
        
        logger.info(f"HRP fitted with {len(self.assets)} assets")
        
        return self
    
    def _get_linkage(self) -> np.ndarray:
        """
        Perform hierarchical clustering.
        
        Returns:
            Linkage matrix
        """
        # Convert correlation to distance
        distance = np.sqrt(0.5 * (1 - self.correlation))
        
        # Ensure symmetry and zero diagonal
        np.fill_diagonal(distance, 0)
        distance = (distance + distance.T) / 2
        
        # Convert to condensed form
        condensed = squareform(distance)
        
        # Hierarchical clustering
        link = linkage(condensed, method=self.linkage_method)
        
        return link
    
    def _get_quasi_diag(self) -> List[int]:
        """
        Sort assets by hierarchical clustering.
        
        Returns:
            Sorted asset indices
        """
        return list(leaves_list(self.linkage_matrix))
    
    def _get_recursive_bisection(self) -> np.ndarray:
        """
        Allocate weights using recursive bisection.
        
        Returns:
            Array of portfolio weights
        """
        n_assets = len(self.assets)
        weights = np.ones(n_assets)
        
        # Initialize clusters
        clusters = [self.sorted_indices]
        
        while len(clusters) > 0:
            # Split each cluster
            new_clusters = []
            
            for cluster in clusters:
                if len(cluster) > 1:
                    # Split cluster in half
                    split = len(cluster) // 2
                    cluster1 = cluster[:split]
                    cluster2 = cluster[split:]
                    
                    # Calculate cluster variances
                    var1 = self._get_cluster_var(cluster1)
                    var2 = self._get_cluster_var(cluster2)
                    
                    # Inverse-variance allocation between clusters
                    alpha = 1 - var1 / (var1 + var2)
                    
                    # Update weights
                    weights[cluster1] *= alpha
                    weights[cluster2] *= (1 - alpha)
                    
                    # Add to next iteration
                    if len(cluster1) > 1:
                        new_clusters.append(cluster1)
                    if len(cluster2) > 1:
                        new_clusters.append(cluster2)
            
            clusters = new_clusters
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
    
    def _get_cluster_var(self, cluster: List[int]) -> float:
        """
        Calculate variance of a cluster.
        
        Uses inverse-variance portfolio within cluster.
        """
        cov = self.covariance[np.ix_(cluster, cluster)]
        
        # Inverse-variance weights within cluster
        inv_diag = 1 / np.diag(cov)
        weights = inv_diag / inv_diag.sum()
        
        # Cluster variance
        var = weights @ cov @ weights
        
        return var
    
    def get_weights(self) -> np.ndarray:
        """Get portfolio weights."""
        if self.weights is None:
            raise ValueError("Model not fitted")
        return self.weights.copy()
    
    def get_allocation_dict(self) -> Dict[str, float]:
        """Get allocation as dictionary."""
        if self.weights is None:
            raise ValueError("Model not fitted")
        return {asset: float(w) for asset, w in zip(self.assets, self.weights)}
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get portfolio statistics."""
        if self.weights is None:
            raise ValueError("Model not fitted")
        
        # Expected return (historical)
        expected_return = (self.returns.mean() @ self.weights) * 252
        
        # Portfolio volatility
        volatility = np.sqrt(self.weights @ (self.covariance * 252) @ self.weights)
        
        # Diversification ratio
        asset_vols = np.sqrt(np.diag(self.covariance) * 252)
        div_ratio = (self.weights @ asset_vols) / volatility
        
        return {
            'expected_return': float(expected_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(expected_return / volatility) if volatility > 0 else 0,
            'diversification_ratio': float(div_ratio)
        }
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about asset clustering."""
        if self.sorted_indices is None:
            raise ValueError("Model not fitted")
        
        sorted_assets = [self.assets[i] for i in self.sorted_indices]
        
        return {
            'sorted_assets': sorted_assets,
            'linkage_matrix': self.linkage_matrix,
            'n_assets': len(self.assets)
        }


class RiskParityModel:
    """
    Standard Risk Parity (Equal Risk Contribution).
    
    Allocates weights so each asset contributes equally to portfolio risk.
    """
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-8):
        """Initialize Risk Parity model."""
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        self.assets: List[str] = []
        self.covariance: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
    
    def fit(self, returns: pd.DataFrame) -> 'RiskParityModel':
        """Fit risk parity model."""
        self.assets = returns.columns.tolist()
        self.covariance = returns.cov().values
        
        self.weights = self._solve_risk_parity()
        
        return self
    
    def _solve_risk_parity(self) -> np.ndarray:
        """
        Solve for risk parity weights using optimization.
        
        Target: Each asset has equal marginal risk contribution.
        """
        n = len(self.assets)
        
        # Initial weights (equal weighted)
        weights = np.ones(n) / n
        
        for iteration in range(self.max_iterations):
            # Portfolio volatility
            port_var = weights @ self.covariance @ weights
            port_vol = np.sqrt(port_var)
            
            # Marginal risk contribution
            mrc = self.covariance @ weights / port_vol
            
            # Risk contribution
            rc = weights * mrc
            
            # Target: equal risk contribution
            target_rc = port_vol / n
            
            # Update weights
            weights_new = weights * (target_rc / rc) ** 0.5
            weights_new = weights_new / weights_new.sum()
            
            # Check convergence
            if np.max(np.abs(weights_new - weights)) < self.tolerance:
                break
            
            weights = weights_new
        
        return weights
    
    def get_weights(self) -> np.ndarray:
        """Get portfolio weights."""
        if self.weights is None:
            raise ValueError("Model not fitted")
        return self.weights.copy()
    
    def get_allocation_dict(self) -> Dict[str, float]:
        """Get allocation as dictionary."""
        if self.weights is None:
            raise ValueError("Model not fitted")
        return {asset: float(w) for asset, w in zip(self.assets, self.weights)}
    
    def get_risk_contributions(self) -> Dict[str, float]:
        """Get risk contribution of each asset."""
        if self.weights is None:
            raise ValueError("Model not fitted")
        
        port_var = self.weights @ self.covariance @ self.weights
        port_vol = np.sqrt(port_var)
        
        mrc = self.covariance @ self.weights / port_vol
        rc = self.weights * mrc
        rc_pct = rc / port_vol * 100
        
        return {asset: float(r) for asset, r in zip(self.assets, rc_pct)}


class MeanVarianceOptimizer:
    """
    Classical Mean-Variance Optimization.
    
    Markowitz portfolio optimization with optional constraints.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        target_return: float = None,
        max_weight: float = 0.4,
        long_only: bool = True
    ):
        """Initialize Mean-Variance optimizer."""
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.max_weight = max_weight
        self.long_only = long_only
        
        self.assets: List[str] = []
        self.expected_returns: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
    
    def fit(
        self,
        returns: pd.DataFrame,
        expected_returns: np.ndarray = None
    ) -> 'MeanVarianceOptimizer':
        """
        Fit mean-variance optimizer.
        
        Args:
            returns: DataFrame of historical returns
            expected_returns: Optional forward-looking returns
        """
        self.assets = returns.columns.tolist()
        self.covariance = returns.cov().values * 252
        
        if expected_returns is not None:
            self.expected_returns = expected_returns
        else:
            self.expected_returns = returns.mean().values * 252
        
        self.weights = self._optimize()
        
        return self
    
    def _optimize(self) -> np.ndarray:
        """
        Solve quadratic program for optimal weights.
        
        Using analytical solution for unconstrained case,
        with post-hoc constraint enforcement.
        """
        n = len(self.assets)
        
        try:
            # Try analytical solution (max Sharpe)
            cov_inv = np.linalg.inv(self.covariance)
            excess_returns = self.expected_returns - self.risk_free_rate
            raw_weights = cov_inv @ excess_returns
            
            # Normalize
            weights = raw_weights / raw_weights.sum()
            
        except np.linalg.LinAlgError:
            # Singular covariance, use equal weights
            logger.warning("Singular covariance matrix, using equal weights")
            weights = np.ones(n) / n
        
        # Apply constraints
        if self.long_only:
            weights = np.maximum(weights, 0)
        
        weights = np.minimum(weights, self.max_weight)
        weights = weights / weights.sum()
        
        return weights
    
    def get_efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier.
        
        Returns:
            DataFrame with returns, volatility, Sharpe for each point
        """
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        results = []
        
        for target in target_returns:
            self.target_return = target
            weights = self._optimize()
            
            port_return = weights @ self.expected_returns
            port_vol = np.sqrt(weights @ self.covariance @ weights)
            sharpe = (port_return - self.risk_free_rate) / port_vol
            
            results.append({
                'return': port_return,
                'volatility': port_vol,
                'sharpe': sharpe
            })
        
        self.target_return = None
        
        return pd.DataFrame(results)
    
    def get_weights(self) -> np.ndarray:
        """Get optimal weights."""
        if self.weights is None:
            raise ValueError("Model not fitted")
        return self.weights.copy()
    
    def get_allocation_dict(self) -> Dict[str, float]:
        """Get allocation as dictionary."""
        if self.weights is None:
            raise ValueError("Model not fitted")
        return {asset: float(w) for asset, w in zip(self.assets, self.weights)}
