"""
Sentiment Analysis Module - NLP-based market sentiment analysis.

Aggregates sentiment from multiple sources:
- Social media (Twitter, Reddit)
- News headlines
- Market commentary

All outputs normalized to [-1, 1] range.
"""

import logging
import re
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np

try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)


@dataclass
class SentimentData:
    """Sentiment analysis result."""
    timestamp: datetime
    source: str
    symbol: str

    # Raw sentiment score (-1 to 1)
    sentiment_score: float

    # Confidence in the sentiment (0 to 1)
    confidence: float

    # Volume/count of analyzed items
    sample_size: int

    # Additional context
    keywords: List[str] = field(default_factory=list)
    extremity: float = 0.0  # How extreme the sentiment is


class SentimentLexicon:
    """
    Simple rule-based sentiment lexicon for financial text.

    Uses word lists with sentiment weights derived from
    financial corpus analysis.
    """

    # Positive financial words with weights
    POSITIVE_WORDS = {
        # Strong positive
        'bullish': 0.8, 'surge': 0.7, 'soar': 0.7, 'rally': 0.6,
        'breakout': 0.6, 'moon': 0.8, 'pump': 0.5, 'gain': 0.4,
        'profit': 0.4, 'growth': 0.4, 'strong': 0.3, 'buy': 0.3,
        'long': 0.3, 'accumulate': 0.4, 'hodl': 0.4, 'bull': 0.5,

        # Moderate positive
        'rise': 0.3, 'up': 0.2, 'high': 0.2, 'green': 0.3,
        'support': 0.2, 'bounce': 0.3, 'recovery': 0.4, 'momentum': 0.3,
        'optimistic': 0.4, 'confident': 0.3, 'opportunity': 0.3,
    }

    # Negative financial words with weights
    NEGATIVE_WORDS = {
        # Strong negative
        'bearish': -0.8, 'crash': -0.8, 'dump': -0.7, 'plunge': -0.7,
        'collapse': -0.8, 'rekt': -0.7, 'fear': -0.5, 'panic': -0.6,
        'sell': -0.3, 'short': -0.3, 'loss': -0.4, 'weak': -0.3,

        # Moderate negative
        'fall': -0.3, 'drop': -0.3, 'down': -0.2, 'low': -0.2,
        'red': -0.3, 'resistance': -0.2, 'decline': -0.4, 'correction': -0.3,
        'pessimistic': -0.4, 'worried': -0.3, 'risk': -0.2, 'bear': -0.5,
    }

    # Intensifiers
    INTENSIFIERS = {
        'very': 1.5, 'extremely': 2.0, 'super': 1.5, 'really': 1.3,
        'absolutely': 1.8, 'definitely': 1.3, 'incredibly': 1.5,
    }

    # Negators (flip sentiment)
    NEGATORS = {'not', 'no', 'never', 'nothing', 'neither', "n't", 'dont', "don't"}

    def analyze_text(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text.

        Returns:
            Tuple of (sentiment_score, confidence)
        """
        if not text:
            return 0.0, 0.0

        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        if not words:
            return 0.0, 0.0

        scores = []
        negated = False
        intensifier = 1.0

        for i, word in enumerate(words):
            # Check for negators
            if word in self.NEGATORS:
                negated = True
                continue

            # Check for intensifiers
            if word in self.INTENSIFIERS:
                intensifier = self.INTENSIFIERS[word]
                continue

            # Get sentiment score
            score = self.POSITIVE_WORDS.get(word, 0) + self.NEGATIVE_WORDS.get(word, 0)

            if score != 0:
                # Apply modifiers
                if negated:
                    score = -score * 0.8  # Negation reduces intensity slightly
                    negated = False

                score *= intensifier
                intensifier = 1.0  # Reset after use

                scores.append(score)
            else:
                # Reset modifiers if not used
                if i > 0:  # Give modifiers a chance to apply
                    negated = False
                    intensifier = 1.0

        if not scores:
            return 0.0, 0.0

        # Calculate final sentiment
        sentiment = np.mean(scores)
        sentiment = np.clip(sentiment, -1, 1)

        # Confidence based on number of sentiment words found
        confidence = min(1.0, len(scores) / 10.0)

        return float(sentiment), float(confidence)


class SentimentAnalyzer:
    """
    Multi-source sentiment analyzer.

    Aggregates sentiment from various sources with proper weighting.
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 600,  # 10 minute cache
        history_size: int = 100
    ):
        """
        Initialize sentiment analyzer.

        Args:
            cache_ttl_seconds: Cache TTL
            history_size: Size of sentiment history for normalization
        """
        self.cache_ttl = cache_ttl_seconds
        self.lexicon = SentimentLexicon()

        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._sentiment_history: Dict[str, deque] = {}
        self._history_size = history_size

    async def analyze_texts(
        self,
        texts: List[str],
        source: str = "unknown",
        symbol: str = "unknown"
    ) -> SentimentData:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of text samples
            source: Source identifier
            symbol: Trading symbol

        Returns:
            SentimentData object
        """
        if not texts:
            return SentimentData(
                timestamp=datetime.now(),
                source=source,
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                sample_size=0
            )

        scores = []
        confidences = []
        keywords = []

        for text in texts:
            score, confidence = self.lexicon.analyze_text(text)
            if confidence > 0:
                scores.append(score)
                confidences.append(confidence)

                # Extract keywords
                keywords.extend(self._extract_keywords(text))

        if not scores:
            return SentimentData(
                timestamp=datetime.now(),
                source=source,
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                sample_size=len(texts)
            )

        # Weighted average by confidence
        weights = np.array(confidences)
        weighted_sentiment = np.average(scores, weights=weights)
        avg_confidence = np.mean(confidences)

        # Calculate extremity
        extremity = np.std(scores) if len(scores) > 1 else 0.0

        # Normalize using history
        normalized_sentiment = self._normalize_sentiment(
            weighted_sentiment, symbol, source
        )

        return SentimentData(
            timestamp=datetime.now(),
            source=source,
            symbol=symbol,
            sentiment_score=float(normalized_sentiment),
            confidence=float(avg_confidence),
            sample_size=len(texts),
            keywords=list(set(keywords))[:10],
            extremity=float(extremity)
        )

    async def get_aggregate_sentiment(
        self,
        symbol: str,
        sources: Dict[str, List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated sentiment from all sources.

        Args:
            symbol: Trading symbol
            sources: Dictionary of source -> texts

        Returns:
            Aggregated sentiment signals
        """
        if sources is None:
            sources = {}

        sentiments = {}
        total_weight = 0.0
        weighted_sum = 0.0

        # Source weights (can be calibrated from historical performance)
        source_weights = {
            'twitter': 0.3,
            'reddit': 0.25,
            'news': 0.35,
            'telegram': 0.1,
        }

        for source, texts in sources.items():
            if texts:
                result = await self.analyze_texts(texts, source, symbol)
                sentiments[source] = {
                    'score': result.sentiment_score,
                    'confidence': result.confidence,
                    'sample_size': result.sample_size,
                    'keywords': result.keywords,
                }

                weight = source_weights.get(source, 0.2) * result.confidence
                weighted_sum += result.sentiment_score * weight
                total_weight += weight

        aggregate = weighted_sum / total_weight if total_weight > 0 else 0.0

        return {
            'aggregate_sentiment': float(np.clip(aggregate, -1, 1)),
            'aggregate_confidence': float(min(1.0, total_weight)),
            'sources': sentiments,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract sentiment-relevant keywords from text."""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        keywords = []
        for word in words:
            if word in self.lexicon.POSITIVE_WORDS or word in self.lexicon.NEGATIVE_WORDS:
                keywords.append(word)

        return keywords

    def _normalize_sentiment(
        self,
        sentiment: float,
        symbol: str,
        source: str
    ) -> float:
        """
        Normalize sentiment using historical distribution.
        """
        key = f"{symbol}_{source}"

        if key not in self._sentiment_history:
            self._sentiment_history[key] = deque(maxlen=self._history_size)

        history = self._sentiment_history[key]
        history.append(sentiment)

        if len(history) < 10:
            # Not enough history, return raw sentiment
            return sentiment

        # Percentile-based normalization
        history_array = np.array(history)
        percentile = np.sum(history_array <= sentiment) / len(history_array)

        # Convert to [-1, 1]
        return float((percentile - 0.5) * 2)


class MockSentimentAnalyzer(SentimentAnalyzer):
    """
    Mock sentiment analyzer for testing.
    """

    async def get_aggregate_sentiment(
        self,
        symbol: str,
        sources: Dict[str, List[str]] = None
    ) -> Dict[str, Any]:
        """Return mock sentiment data."""
        import random

        sentiment = random.gauss(0, 0.3)
        sentiment = np.clip(sentiment, -1, 1)

        return {
            'aggregate_sentiment': float(sentiment),
            'aggregate_confidence': random.uniform(0.5, 0.9),
            'sources': {
                'twitter': {'score': random.uniform(-1, 1), 'confidence': 0.7},
                'reddit': {'score': random.uniform(-1, 1), 'confidence': 0.6},
            },
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
        }


class SocialMediaScraper:
    """
    Base class for social media scrapers.

    Note: In production, use official APIs with proper authentication.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize scraper."""
        self.api_key = api_key

    async def fetch_posts(
        self,
        query: str,
        limit: int = 100,
        since: datetime = None
    ) -> List[str]:
        """
        Fetch posts matching query.

        Override in subclasses for specific platforms.
        """
        raise NotImplementedError


class TwitterSentiment(SocialMediaScraper):
    """
    Twitter sentiment fetcher.

    Uses Twitter API v2 for searching tweets.
    """

    async def fetch_posts(
        self,
        query: str,
        limit: int = 100,
        since: datetime = None
    ) -> List[str]:
        """
        Fetch tweets matching query.

        Args:
            query: Search query
            limit: Maximum tweets to fetch
            since: Minimum tweet time

        Returns:
            List of tweet texts
        """
        # In production, implement Twitter API v2 calls
        # Requires Bearer Token authentication
        logger.warning("Twitter API not configured - returning empty list")
        return []


class RedditSentiment(SocialMediaScraper):
    """
    Reddit sentiment fetcher.

    Uses Reddit API for fetching posts and comments.
    """

    CRYPTO_SUBREDDITS = [
        'cryptocurrency', 'bitcoin', 'ethereum', 'cryptomarkets'
    ]
    FOREX_SUBREDDITS = [
        'forex', 'forextrading', 'daytrading'
    ]

    async def fetch_posts(
        self,
        query: str,
        limit: int = 100,
        since: datetime = None
    ) -> List[str]:
        """
        Fetch Reddit posts matching query.

        Args:
            query: Search query or subreddit
            limit: Maximum posts to fetch
            since: Minimum post time

        Returns:
            List of post texts
        """
        # In production, implement Reddit API calls
        # Requires OAuth2 authentication
        logger.warning("Reddit API not configured - returning empty list")
        return []
