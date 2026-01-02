"""
Alternative.me Fear & Greed Index API Client.

The Fear & Greed Index is a contrarian indicator:
- Extreme Fear (0-25) = Buying opportunity
- Extreme Greed (75-100) = Selling opportunity
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class FearGreedData:
    """Fear & Greed Index data point."""
    value: int  # 0-100
    value_classification: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: datetime
    
    @property
    def signal_direction(self) -> float:
        """
        Convert to trading signal.
        
        Extreme Fear → Buy (1.0)
        Extreme Greed → Sell (-1.0)
        """
        # Normalize to -1 to 1 (inverted because contrarian)
        return (50 - self.value) / 50
    
    @property
    def is_extreme(self) -> bool:
        """Check if in extreme zone."""
        return self.value <= 25 or self.value >= 75


class FearGreedClient:
    """
    Client for Alternative.me Fear & Greed Index API.
    
    Usage:
        client = FearGreedClient()
        current = client.get_current()
        print(f"Fear & Greed: {current.value} ({current.value_classification})")
        
        # Get historical data
        history = client.get_historical(days=30)
    """
    
    BASE_URL = "https://api.alternative.me/fng/"
    
    def __init__(self):
        """Initialize Fear & Greed client."""
        self.session = requests.Session()
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=30)
    
    def _make_request(self, params: Dict = None) -> Dict:
        """Make API request."""
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Fear & Greed API request failed: {e}")
            return {}
    
    def get_current(self) -> Optional[FearGreedData]:
        """
        Get current Fear & Greed Index value.
        
        Returns:
            FearGreedData object or None on error
        """
        # Check cache
        if self._cache_time and datetime.now() - self._cache_time < self._cache_ttl:
            if 'current' in self._cache:
                return self._cache['current']
        
        data = self._make_request({'limit': 1})
        
        if not data or 'data' not in data:
            return None
        
        try:
            item = data['data'][0]
            result = FearGreedData(
                value=int(item['value']),
                value_classification=item['value_classification'],
                timestamp=datetime.fromtimestamp(int(item['timestamp']))
            )
            
            # Cache result
            self._cache['current'] = result
            self._cache_time = datetime.now()
            
            return result
            
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error parsing Fear & Greed data: {e}")
            return None
    
    def get_historical(self, days: int = 30) -> List[FearGreedData]:
        """
        Get historical Fear & Greed Index values.
        
        Args:
            days: Number of days of history
            
        Returns:
            List of FearGreedData objects (newest first)
        """
        data = self._make_request({'limit': days})
        
        if not data or 'data' not in data:
            return []
        
        results = []
        
        for item in data['data']:
            try:
                fg = FearGreedData(
                    value=int(item['value']),
                    value_classification=item['value_classification'],
                    timestamp=datetime.fromtimestamp(int(item['timestamp']))
                )
                results.append(fg)
            except (KeyError, ValueError) as e:
                logger.warning(f"Error parsing historical data: {e}")
        
        return results
    
    def get_signal(self) -> Dict[str, Any]:
        """
        Get trading signal from Fear & Greed Index.
        
        Returns:
            Dict with direction, confidence, and raw value
        """
        current = self.get_current()
        
        if not current:
            return {
                'direction': 0.0,
                'confidence': 0.0,
                'value': 50,
                'classification': 'Unknown',
                'timestamp': datetime.now()
            }
        
        # Calculate signal direction
        direction = current.signal_direction
        
        # Confidence based on extremity
        if current.value <= 10 or current.value >= 90:
            confidence = 0.9
        elif current.value <= 20 or current.value >= 80:
            confidence = 0.7
        elif current.value <= 25 or current.value >= 75:
            confidence = 0.5
        elif current.value <= 30 or current.value >= 70:
            confidence = 0.3
        else:
            confidence = 0.1
        
        return {
            'direction': direction,
            'confidence': confidence,
            'value': current.value,
            'classification': current.value_classification,
            'is_extreme': current.is_extreme,
            'timestamp': current.timestamp
        }
    
    def get_trend(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze Fear & Greed trend over recent days.
        
        Returns:
            Dict with trend direction and momentum
        """
        history = self.get_historical(days)
        
        if len(history) < 2:
            return {'trend': 0.0, 'momentum': 0.0}
        
        values = [fg.value for fg in history]
        
        # Current vs average
        current = values[0]
        avg = sum(values) / len(values)
        
        # Trend: positive if fear increasing (bullish contrarian)
        # negative if greed increasing (bearish contrarian)
        trend = (avg - current) / 50  # Normalize
        
        # Momentum: rate of change
        if len(values) >= 3:
            recent_change = values[0] - values[2]
            momentum = -recent_change / 100  # Contrarian
        else:
            momentum = 0.0
        
        return {
            'trend': max(-1, min(1, trend)),
            'momentum': max(-1, min(1, momentum)),
            'current': current,
            'average': avg,
            'min': min(values),
            'max': max(values)
        }


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources.
    
    Sources:
    - Fear & Greed Index
    - Funding rates
    - Long/short ratios
    - Social sentiment (if available)
    """
    
    def __init__(
        self,
        fear_greed_client: Optional[FearGreedClient] = None,
        coinglass_client = None
    ):
        """
        Initialize sentiment aggregator.
        
        Args:
            fear_greed_client: FearGreedClient instance
            coinglass_client: CoinglassClient instance
        """
        self.fear_greed = fear_greed_client or FearGreedClient()
        self.coinglass = coinglass_client
    
    def get_aggregate_sentiment(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get aggregated sentiment signal.
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Dict with aggregate direction and confidence
        """
        signals = []
        weights = []
        
        # Fear & Greed signal
        fg_signal = self.fear_greed.get_signal()
        if fg_signal['confidence'] > 0:
            signals.append(fg_signal['direction'])
            weights.append(fg_signal['confidence'] * 0.4)  # 40% weight
        
        # Funding rate signal (if available)
        if self.coinglass:
            try:
                funding_signal = self.coinglass.get_funding_signal(symbol)
                if funding_signal['confidence'] > 0:
                    signals.append(funding_signal['direction'])
                    weights.append(funding_signal['confidence'] * 0.4)
            except Exception as e:
                logger.warning(f"Error getting funding signal: {e}")
        
        # Calculate weighted average
        if not signals:
            return {
                'direction': 0.0,
                'confidence': 0.0,
                'sources': []
            }
        
        total_weight = sum(weights)
        weighted_direction = sum(s * w for s, w in zip(signals, weights)) / total_weight
        avg_confidence = total_weight / len(signals)
        
        return {
            'direction': weighted_direction,
            'confidence': avg_confidence,
            'fear_greed': fg_signal,
            'sources': ['fear_greed', 'funding'] if self.coinglass else ['fear_greed'],
            'timestamp': datetime.now()
        }
