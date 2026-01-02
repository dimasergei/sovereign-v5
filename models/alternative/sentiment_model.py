"""
Sentiment NLP Model - Social sentiment analysis for trading signals.

Analyzes text from social media, news, and other sources to
generate trading signals based on market sentiment.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import re

logger = logging.getLogger(__name__)

# Try importing NLP libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel, ModelPrediction


class SentimentAnalyzer:
    """
    Multi-method sentiment analyzer.
    
    Supports:
    - TextBlob (fast, simple)
    - FinBERT (accurate, slower)
    - Keyword-based (fastest)
    """
    
    # Crypto-specific keywords
    BULLISH_KEYWORDS = [
        'moon', 'pump', 'bull', 'buy', 'long', 'breakout', 'ath', 'hodl',
        'accumulate', 'bullish', 'green', 'rocket', 'lambo', 'gains',
        'support', 'bounce', 'recovery', 'rally', 'surge'
    ]
    
    BEARISH_KEYWORDS = [
        'dump', 'crash', 'bear', 'sell', 'short', 'breakdown', 'rekt',
        'bearish', 'red', 'blood', 'fear', 'panic', 'resistance',
        'drop', 'fall', 'plunge', 'collapse', 'correction'
    ]
    
    def __init__(self, method: str = 'textblob'):
        """
        Initialize sentiment analyzer.
        
        Args:
            method: 'textblob', 'finbert', or 'keyword'
        """
        self.method = method
        self.finbert_pipeline = None
        
        if method == 'finbert' and TRANSFORMERS_AVAILABLE:
            try:
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert"
                )
            except Exception as e:
                logger.warning(f"FinBERT load failed: {e}, falling back to TextBlob")
                self.method = 'textblob'
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.
        
        Returns:
            Dict with sentiment (-1 to 1) and confidence (0 to 1)
        """
        if not text or not text.strip():
            return {'sentiment': 0.0, 'confidence': 0.0}
        
        # Clean text
        text = self._clean_text(text)
        
        if self.method == 'finbert' and self.finbert_pipeline:
            return self._analyze_finbert(text)
        elif self.method == 'textblob' and TEXTBLOB_AVAILABLE:
            return self._analyze_textblob(text)
        else:
            return self._analyze_keywords(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags (keep the word)
        text = re.sub(r'#(\w+)', r'\1', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    def _analyze_textblob(self, text: str) -> Dict[str, float]:
        """Analyze using TextBlob."""
        blob = TextBlob(text)
        
        # TextBlob polarity is -1 to 1
        sentiment = blob.sentiment.polarity
        
        # Subjectivity as proxy for confidence
        confidence = blob.sentiment.subjectivity * 0.5 + 0.5
        
        return {'sentiment': sentiment, 'confidence': confidence}
    
    def _analyze_finbert(self, text: str) -> Dict[str, float]:
        """Analyze using FinBERT."""
        try:
            result = self.finbert_pipeline(text[:512])[0]  # Max 512 tokens
            
            label = result['label']
            score = result['score']
            
            if label == 'positive':
                sentiment = score
            elif label == 'negative':
                sentiment = -score
            else:
                sentiment = 0.0
            
            return {'sentiment': sentiment, 'confidence': score}
            
        except Exception as e:
            logger.warning(f"FinBERT error: {e}")
            return self._analyze_keywords(text)
    
    def _analyze_keywords(self, text: str) -> Dict[str, float]:
        """Analyze using keyword matching."""
        words = text.lower().split()
        
        bullish_count = sum(1 for w in words if w in self.BULLISH_KEYWORDS)
        bearish_count = sum(1 for w in words if w in self.BEARISH_KEYWORDS)
        
        total = bullish_count + bearish_count
        
        if total == 0:
            return {'sentiment': 0.0, 'confidence': 0.0}
        
        sentiment = (bullish_count - bearish_count) / total
        confidence = min(1.0, total / 10)  # More keywords = more confidence
        
        return {'sentiment': sentiment, 'confidence': confidence}
    
    def analyze_batch(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze batch of texts and return aggregate sentiment.
        """
        if not texts:
            return {'sentiment': 0.0, 'confidence': 0.0, 'count': 0}
        
        sentiments = []
        confidences = []
        
        for text in texts:
            result = self.analyze(text)
            if result['confidence'] > 0:
                sentiments.append(result['sentiment'])
                confidences.append(result['confidence'])
        
        if not sentiments:
            return {'sentiment': 0.0, 'confidence': 0.0, 'count': 0}
        
        # Weighted average by confidence
        weights = np.array(confidences)
        weights = weights / weights.sum()
        
        avg_sentiment = np.average(sentiments, weights=weights)
        avg_confidence = np.mean(confidences)
        
        return {
            'sentiment': float(avg_sentiment),
            'confidence': float(avg_confidence),
            'count': len(sentiments),
            'std': float(np.std(sentiments)) if len(sentiments) > 1 else 0.0
        }


class SentimentModel(BaseModel):
    """
    Sentiment-based trading model.
    
    Generates signals from social media sentiment with
    contrarian and momentum modes.
    
    Usage:
        model = SentimentModel(mode='contrarian')
        
        # Analyze texts
        signal = model.predict_from_texts(tweets)
        
        # Or use pre-computed sentiment
        signal = model.predict(sentiment_score)
    """
    
    def __init__(
        self,
        name: str = "sentiment",
        mode: str = 'contrarian',  # 'contrarian' or 'momentum'
        extreme_threshold: float = 0.6,
        analyzer_method: str = 'textblob'
    ):
        """
        Initialize sentiment model.
        
        Args:
            name: Model name
            mode: 'contrarian' (fade extremes) or 'momentum' (follow sentiment)
            extreme_threshold: Threshold for extreme sentiment
            analyzer_method: Sentiment analysis method
        """
        super().__init__(name)
        
        self.mode = mode
        self.extreme_threshold = extreme_threshold
        self.analyzer = SentimentAnalyzer(method=analyzer_method)
        
        # Calibrated parameters
        self.sentiment_lag: int = 1  # Hours of lag before price follows
        self.decay_rate: float = 0.8
    
    def fit(
        self,
        sentiments: np.ndarray,
        returns: np.ndarray,
        lookahead: int = 24
    ) -> 'SentimentModel':
        """
        Fit model to historical sentiment and returns.
        
        Determines optimal mode (contrarian vs momentum) and thresholds.
        """
        # Test contrarian vs momentum
        contrarian_accuracy = 0
        momentum_accuracy = 0
        
        future_returns = np.roll(returns, -lookahead)
        future_returns[-lookahead:] = 0
        
        for threshold in np.arange(0.3, 0.9, 0.1):
            # Contrarian signals
            contrarian = np.where(
                sentiments > threshold, -1,
                np.where(sentiments < -threshold, 1, 0)
            )
            
            # Momentum signals
            momentum = np.where(
                sentiments > threshold, 1,
                np.where(sentiments < -threshold, -1, 0)
            )
            
            # Calculate accuracies
            c_correct = (contrarian * future_returns > 0)[contrarian != 0].mean()
            m_correct = (momentum * future_returns > 0)[momentum != 0].mean()
            
            if c_correct > contrarian_accuracy:
                contrarian_accuracy = c_correct
            if m_correct > momentum_accuracy:
                momentum_accuracy = m_correct
        
        # Choose better mode
        if contrarian_accuracy > momentum_accuracy:
            self.mode = 'contrarian'
            best_accuracy = contrarian_accuracy
        else:
            self.mode = 'momentum'
            best_accuracy = momentum_accuracy
        
        self.is_trained = True
        self.training_date = datetime.now()
        
        self.metadata = {
            'mode': self.mode,
            'accuracy': best_accuracy,
            'contrarian_acc': contrarian_accuracy,
            'momentum_acc': momentum_accuracy
        }
        
        logger.info(f"Sentiment model fitted: mode={self.mode}, accuracy={best_accuracy:.2%}")
        
        return self
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        Generate prediction from sentiment score.
        
        Args:
            X: Sentiment score (-1 to 1)
        """
        sentiment = float(X.flatten()[-1]) if hasattr(X, 'flatten') else float(X)
        
        return self._generate_signal(sentiment, confidence=0.7)
    
    def predict_from_texts(self, texts: List[str]) -> ModelPrediction:
        """
        Generate prediction from raw texts.
        
        Args:
            texts: List of text strings to analyze
        """
        result = self.analyzer.analyze_batch(texts)
        
        return self._generate_signal(
            result['sentiment'],
            confidence=result['confidence']
        )
    
    def _generate_signal(
        self,
        sentiment: float,
        confidence: float
    ) -> ModelPrediction:
        """Generate trading signal from sentiment."""
        
        # Apply mode
        if self.mode == 'contrarian':
            # Fade extreme sentiment
            if sentiment > self.extreme_threshold:
                direction = -1.0
            elif sentiment < -self.extreme_threshold:
                direction = 1.0
            elif abs(sentiment) > self.extreme_threshold * 0.5:
                direction = -np.sign(sentiment) * 0.5
            else:
                direction = 0.0
        else:
            # Follow sentiment
            if abs(sentiment) > self.extreme_threshold:
                direction = np.sign(sentiment)
            elif abs(sentiment) > self.extreme_threshold * 0.5:
                direction = np.sign(sentiment) * 0.5
            else:
                direction = 0.0
        
        # Confidence based on sentiment extremity and input confidence
        signal_confidence = confidence * min(1.0, abs(sentiment) / self.extreme_threshold)
        
        return ModelPrediction(
            model_name=self.name,
            direction=direction,
            magnitude=abs(sentiment) * 0.01,
            confidence=signal_confidence,
            metadata={
                'sentiment': sentiment,
                'mode': self.mode,
                'is_extreme': abs(sentiment) > self.extreme_threshold
            }
        )
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'extreme_threshold': self.extreme_threshold,
            'sentiment_lag': self.sentiment_lag,
            'decay_rate': self.decay_rate
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.mode = state['mode']
        self.extreme_threshold = state['extreme_threshold']
        self.sentiment_lag = state['sentiment_lag']
        self.decay_rate = state['decay_rate']


class SocialMediaCollector:
    """
    Collects and processes social media data.
    
    Note: Requires API keys for actual collection.
    This is a framework that can be extended with real APIs.
    """
    
    def __init__(self):
        """Initialize collector."""
        self.cache: Dict[str, List[Dict]] = {}
    
    def collect_twitter(
        self,
        query: str,
        count: int = 100
    ) -> List[str]:
        """
        Collect tweets (placeholder for Twitter API integration).
        
        In production, use tweepy with Twitter API credentials.
        """
        logger.warning("Twitter collection not implemented - need API keys")
        return []
    
    def collect_reddit(
        self,
        subreddit: str,
        count: int = 100
    ) -> List[str]:
        """
        Collect Reddit posts (placeholder for Reddit API integration).
        
        In production, use praw with Reddit API credentials.
        """
        logger.warning("Reddit collection not implemented - need API keys")
        return []
    
    def get_crypto_sentiment(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get aggregated crypto sentiment.
        
        Combines multiple sources when available.
        """
        texts = []
        
        # Collect from available sources
        texts.extend(self.collect_twitter(f"${symbol}"))
        texts.extend(self.collect_reddit(f"r/cryptocurrency {symbol}"))
        
        if not texts:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'source': 'none',
                'warning': 'No data collected - API keys needed'
            }
        
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_batch(texts)
        result['source'] = 'social_media'
        
        return result
