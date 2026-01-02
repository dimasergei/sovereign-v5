"""
Coinglass API Client - Fetches funding rates and open interest data.

Funding rates are critical for crypto trading as they indicate market sentiment
and can be used for funding rate arbitrage strategies.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

import aiohttp
import requests

logger = logging.getLogger(__name__)


@dataclass
class FundingRate:
    """Funding rate data for a symbol."""
    symbol: str
    exchange: str
    rate: float  # Funding rate as decimal (e.g., 0.0001 = 0.01%)
    predicted_rate: Optional[float]
    next_funding_time: datetime
    timestamp: datetime
    
    @property
    def annualized_rate(self) -> float:
        """Convert to annualized percentage."""
        # Funding typically every 8 hours = 3x/day = 1095x/year
        return self.rate * 1095 * 100


@dataclass
class OpenInterest:
    """Open interest data."""
    symbol: str
    exchange: str
    open_interest: float  # In USD
    open_interest_change_24h: float  # Percentage
    timestamp: datetime


class CoinglassClient:
    """
    Client for Coinglass API.
    
    Provides:
    - Funding rates across exchanges
    - Open interest data
    - Long/short ratios
    - Liquidation data
    
    Usage:
        client = CoinglassClient(api_key="your_key")
        rates = client.get_funding_rates("BTC")
        
        # Get average funding rate
        avg_rate = client.get_average_funding_rate("BTC")
    """
    
    BASE_URL = "https://open-api.coinglass.com/public/v2"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Coinglass client.
        
        Args:
            api_key: Coinglass API key (optional for some endpoints)
        """
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers['coinglassSecret'] = api_key
        
        self._rate_limit_remaining = 100
        self._rate_limit_reset = datetime.now()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting."""
        # Simple rate limiting
        if self._rate_limit_remaining <= 0:
            sleep_time = (self._rate_limit_reset - datetime.now()).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('success') is False:
                logger.error(f"Coinglass API error: {data.get('msg')}")
                return {}
            
            return data.get('data', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Coinglass request failed: {e}")
            return {}
    
    def get_funding_rates(self, symbol: str = "BTC") -> List[FundingRate]:
        """
        Get current funding rates across exchanges.
        
        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
            
        Returns:
            List of FundingRate objects
        """
        data = self._make_request("funding", {"symbol": symbol})
        
        if not data:
            return []
        
        rates = []
        
        for item in data:
            try:
                rate = FundingRate(
                    symbol=symbol,
                    exchange=item.get('exchangeName', 'unknown'),
                    rate=float(item.get('rate', 0)),
                    predicted_rate=float(item.get('predictedRate', 0)) if item.get('predictedRate') else None,
                    next_funding_time=datetime.fromtimestamp(item.get('nextFundingTime', 0) / 1000),
                    timestamp=datetime.now()
                )
                rates.append(rate)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing funding rate: {e}")
        
        return rates
    
    def get_average_funding_rate(self, symbol: str = "BTC") -> float:
        """
        Get average funding rate across major exchanges.
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Average funding rate as decimal
        """
        rates = self.get_funding_rates(symbol)
        
        if not rates:
            return 0.0
        
        # Weight by exchange importance (simplified)
        major_exchanges = ['Binance', 'OKX', 'Bybit', 'Bitget']
        
        major_rates = [r.rate for r in rates if r.exchange in major_exchanges]
        
        if major_rates:
            return sum(major_rates) / len(major_rates)
        
        return sum(r.rate for r in rates) / len(rates)
    
    def get_open_interest(self, symbol: str = "BTC") -> List[OpenInterest]:
        """
        Get open interest data across exchanges.
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            List of OpenInterest objects
        """
        data = self._make_request("open_interest", {"symbol": symbol})
        
        if not data:
            return []
        
        oi_list = []
        
        for item in data:
            try:
                oi = OpenInterest(
                    symbol=symbol,
                    exchange=item.get('exchangeName', 'unknown'),
                    open_interest=float(item.get('openInterest', 0)),
                    open_interest_change_24h=float(item.get('h24Change', 0)),
                    timestamp=datetime.now()
                )
                oi_list.append(oi)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing open interest: {e}")
        
        return oi_list
    
    def get_total_open_interest(self, symbol: str = "BTC") -> float:
        """Get total open interest across all exchanges."""
        oi_list = self.get_open_interest(symbol)
        return sum(oi.open_interest for oi in oi_list)
    
    def get_long_short_ratio(self, symbol: str = "BTC") -> Dict[str, float]:
        """
        Get long/short ratio data.
        
        Returns dict with:
        - long_ratio: Percentage of longs
        - short_ratio: Percentage of shorts
        - long_short_ratio: Longs / Shorts
        """
        data = self._make_request("long_short", {"symbol": symbol})
        
        if not data:
            return {'long_ratio': 50.0, 'short_ratio': 50.0, 'long_short_ratio': 1.0}
        
        # Aggregate across exchanges
        long_accounts = 0
        short_accounts = 0
        
        for item in data:
            long_accounts += float(item.get('longAccount', 0))
            short_accounts += float(item.get('shortAccount', 0))
        
        total = long_accounts + short_accounts
        
        if total == 0:
            return {'long_ratio': 50.0, 'short_ratio': 50.0, 'long_short_ratio': 1.0}
        
        return {
            'long_ratio': long_accounts / total * 100,
            'short_ratio': short_accounts / total * 100,
            'long_short_ratio': long_accounts / short_accounts if short_accounts > 0 else 1.0
        }
    
    def get_liquidations(self, symbol: str = "BTC", hours: int = 24) -> Dict[str, float]:
        """
        Get liquidation data.
        
        Returns:
            Dict with long_liquidations and short_liquidations in USD
        """
        data = self._make_request("liquidation", {"symbol": symbol, "interval": f"{hours}h"})
        
        if not data:
            return {'long_liquidations': 0.0, 'short_liquidations': 0.0}
        
        return {
            'long_liquidations': float(data.get('longLiquidation', 0)),
            'short_liquidations': float(data.get('shortLiquidation', 0))
        }
    
    def get_funding_signal(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get aggregated funding rate signal.
        
        High positive funding = overleveraged longs = bearish
        High negative funding = overleveraged shorts = bullish
        
        Returns:
            Dict with signal direction and strength
        """
        avg_rate = self.get_average_funding_rate(symbol)
        ls_ratio = self.get_long_short_ratio(symbol)
        
        # Annualized rate
        annual_rate = avg_rate * 1095 * 100
        
        # Signal calculation
        # Very high funding (>50% annual) = contrarian short signal
        # Very low funding (<-50% annual) = contrarian long signal
        
        if annual_rate > 100:
            direction = -1.0  # Strong short signal
            confidence = min(1.0, annual_rate / 200)
        elif annual_rate > 50:
            direction = -0.5
            confidence = annual_rate / 100
        elif annual_rate < -100:
            direction = 1.0  # Strong long signal
            confidence = min(1.0, abs(annual_rate) / 200)
        elif annual_rate < -50:
            direction = 0.5
            confidence = abs(annual_rate) / 100
        else:
            direction = 0.0
            confidence = 0.0
        
        # Adjust by long/short ratio
        if ls_ratio['long_short_ratio'] > 2:
            direction -= 0.2  # Too many longs
        elif ls_ratio['long_short_ratio'] < 0.5:
            direction += 0.2  # Too many shorts
        
        return {
            'direction': max(-1, min(1, direction)),
            'confidence': min(1.0, confidence),
            'funding_rate': avg_rate,
            'annualized_rate': annual_rate,
            'long_short_ratio': ls_ratio['long_short_ratio'],
            'timestamp': datetime.now()
        }


class MockCoinglassClient(CoinglassClient):
    """Mock client for testing without API key."""
    
    def __init__(self):
        super().__init__(api_key=None)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Return mock data."""
        import random
        
        if 'funding' in endpoint:
            return [
                {'exchangeName': 'Binance', 'rate': random.uniform(-0.001, 0.001)},
                {'exchangeName': 'OKX', 'rate': random.uniform(-0.001, 0.001)},
            ]
        elif 'open_interest' in endpoint:
            return [
                {'exchangeName': 'Binance', 'openInterest': random.uniform(1e9, 5e9)},
            ]
        elif 'long_short' in endpoint:
            return [
                {'longAccount': random.uniform(40, 60), 'shortAccount': random.uniform(40, 60)},
            ]
        
        return {}
