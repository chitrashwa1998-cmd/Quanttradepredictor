import json
import logging
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from pydantic import BaseModel


class MarketAnalysis(BaseModel):
    """Market analysis response structure"""
    sentiment_score: float  # -1 to 1 scale
    confidence: float  # 0 to 1
    key_factors: List[str]
    recommendation: str
    risk_level: str


class TradingInsight(BaseModel):
    """Trading insight response structure"""
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0 to 1
    time_horizon: str  # "short", "medium", "long"
    reasoning: List[str]


class GeminiAnalyzer:
    """Gemini AI integration for trading analysis"""
    
    def __init__(self):
        """Initialize Gemini client"""
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            self.client = genai.Client(api_key=api_key)
            self.logger = logging.getLogger(__name__)
            print("✅ Gemini AI client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def analyze_market_data(self, ohlc_data: pd.DataFrame, predictions: Dict) -> MarketAnalysis:
        """
        Analyze market data with current predictions using Gemini AI
        
        Args:
            ohlc_data: Recent OHLC data
            predictions: Current ML model predictions
        
        Returns:
            MarketAnalysis with sentiment and insights
        """
        try:
            # Prepare market data summary
            recent_data = ohlc_data.tail(20)  # Last 20 candles
            
            market_summary = {
                "current_price": float(recent_data['close'].iloc[-1]),
                "price_change_5": float(((recent_data['close'].iloc[-1] / recent_data['close'].iloc[-6]) - 1) * 100),
                "price_change_20": float(((recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1) * 100),
                "volume_trend": "increasing" if recent_data['volume'].iloc[-5:].mean() > recent_data['volume'].iloc[-10:-5].mean() else "decreasing",
                "volatility": float(recent_data['close'].std()),
                "high_low_range": float(recent_data['high'].max() - recent_data['low'].min())
            }
            
            # Create analysis prompt
            prompt = f"""
            Analyze the following Nifty 50 index market data and ML predictions:
            
            **Current Market State:**
            - Current Price: {market_summary['current_price']:.2f}
            - 5-period change: {market_summary['price_change_5']:.2f}%
            - 20-period change: {market_summary['price_change_20']:.2f}%
            - Volume trend: {market_summary['volume_trend']}
            - Recent volatility: {market_summary['volatility']:.2f}
            
            **ML Model Predictions:**
            {json.dumps(predictions, indent=2)}
            
            Provide a comprehensive market analysis with:
            1. Sentiment score (-1 to 1, where -1 is very bearish, 1 is very bullish)
            2. Confidence level (0 to 1)
            3. Key market factors driving the analysis
            4. Trading recommendation
            5. Risk assessment (low, medium, high)
            
            Focus on how the technical indicators align with current market conditions.
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert quantitative analyst specializing in Indian stock market analysis. Provide precise, data-driven insights.",
                    response_mime_type="application/json",
                    response_schema=MarketAnalysis,
                    temperature=0.3
                )
            )
            
            if response.text:
                analysis_data = json.loads(response.text)
                return MarketAnalysis(**analysis_data)
            else:
                raise ValueError("Empty response from Gemini")
                
        except Exception as e:
            self.logger.error(f"Market analysis failed: {e}")
            # Return default analysis
            return MarketAnalysis(
                sentiment_score=0.0,
                confidence=0.5,
                key_factors=["Technical analysis only"],
                recommendation="Hold - AI analysis unavailable",
                risk_level="medium"
            )
    
    def generate_trading_insights(self, market_data: pd.DataFrame, model_predictions: Dict) -> TradingInsight:
        """
        Generate specific trading insights based on data and predictions
        
        Args:
            market_data: OHLC data
            model_predictions: ML model outputs
        
        Returns:
            TradingInsight with direction and reasoning
        """
        try:
            recent_candles = market_data.tail(10)
            
            # Extract key metrics
            price_momentum = ((recent_candles['close'].iloc[-1] / recent_candles['close'].iloc[0]) - 1) * 100
            volume_strength = recent_candles['volume'].iloc[-3:].mean() / recent_candles['volume'].mean()
            
            prompt = f"""
            Based on the following Nifty 50 trading data and ML predictions, provide specific trading insights:
            
            **Recent Performance:**
            - 10-period momentum: {price_momentum:.2f}%
            - Volume strength ratio: {volume_strength:.2f}
            - Current volatility prediction: {model_predictions.get('volatility', 'N/A')}
            - Direction prediction: {model_predictions.get('direction', 'N/A')}
            - Profit probability: {model_predictions.get('profit_probability', 'N/A')}
            - Reversal signal: {model_predictions.get('reversal', 'N/A')}
            
            **Latest Price Action:**
            - Open: {recent_candles['open'].iloc[-1]:.2f}
            - High: {recent_candles['high'].iloc[-1]:.2f}
            - Low: {recent_candles['low'].iloc[-1]:.2f}
            - Close: {recent_candles['close'].iloc[-1]:.2f}
            
            Provide:
            1. Overall direction (bullish/bearish/neutral)
            2. Signal strength (0-1 scale)
            3. Time horizon (short/medium/long term)
            4. Key reasoning points for the recommendation
            
            Consider both technical patterns and ML predictions in your analysis.
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a professional trading advisor. Provide actionable insights based on technical and quantitative analysis.",
                    response_mime_type="application/json",
                    response_schema=TradingInsight,
                    temperature=0.2
                )
            )
            
            if response.text:
                insight_data = json.loads(response.text)
                return TradingInsight(**insight_data)
            else:
                raise ValueError("Empty response from Gemini")
                
        except Exception as e:
            self.logger.error(f"Trading insight generation failed: {e}")
            return TradingInsight(
                direction="neutral",
                strength=0.5,
                time_horizon="short",
                reasoning=["AI analysis unavailable", "Rely on ML predictions only"]
            )
    
    def explain_model_predictions(self, predictions: Dict, feature_importance: Optional[Dict] = None) -> str:
        """
        Generate natural language explanation of ML model predictions
        
        Args:
            predictions: Model prediction results
            feature_importance: Optional feature importance data
        
        Returns:
            Natural language explanation
        """
        try:
            prompt = f"""
            Explain these machine learning model predictions for Nifty 50 trading in simple terms:
            
            **Predictions:**
            {json.dumps(predictions, indent=2)}
            
            **Feature Importance (if available):**
            {json.dumps(feature_importance, indent=2) if feature_importance else "Not available"}
            
            Provide a clear, concise explanation that:
            1. Summarizes what each prediction means
            2. Explains the confidence levels
            3. Identifies the most important factors
            4. Gives practical trading implications
            
            Write in a way that both novice and experienced traders can understand.
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4
                )
            )
            
            return response.text or "Model explanation unavailable"
            
        except Exception as e:
            self.logger.error(f"Prediction explanation failed: {e}")
            return "Unable to generate model explanation at this time."
    
    def analyze_risk_factors(self, market_data: pd.DataFrame, predictions: Dict) -> Dict:
        """
        Analyze current risk factors in the market
        
        Args:
            market_data: OHLC data
            predictions: Model predictions
        
        Returns:
            Risk analysis dictionary
        """
        try:
            recent_data = market_data.tail(30)
            
            # Calculate risk metrics
            volatility_30d = recent_data['close'].pct_change().std() * 100
            max_drawdown = ((recent_data['close'].cummax() - recent_data['close']) / recent_data['close'].cummax()).max() * 100
            
            prompt = f"""
            Analyze the risk factors for Nifty 50 trading based on:
            
            **Risk Metrics:**
            - 30-day volatility: {volatility_30d:.2f}%
            - Maximum drawdown: {max_drawdown:.2f}%
            - Model predictions: {json.dumps(predictions)}
            
            **Current Market State:**
            - Price range (30d): {recent_data['low'].min():.2f} - {recent_data['high'].max():.2f}
            - Average volume: {recent_data['volume'].mean():.0f}
            
            Identify and assess:
            1. Primary risk factors
            2. Risk level (1-10 scale)
            3. Recommended position sizing
            4. Stop-loss suggestions
            5. Time-based risks (intraday, swing, positional)
            
            Provide practical risk management advice.
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )
            
            # Parse response into structured format
            risk_text = response.text or "Risk analysis unavailable"
            
            return {
                "analysis": risk_text,
                "volatility_score": min(volatility_30d / 2, 10),  # Scale to 0-10
                "drawdown_risk": max_drawdown,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Risk analysis failed: {e}")
            return {
                "analysis": "Risk analysis currently unavailable",
                "volatility_score": 5.0,
                "drawdown_risk": 0.0,
                "timestamp": datetime.now().isoformat()
            }


def test_gemini_connection() -> bool:
    """Test Gemini API connection"""
    try:
        analyzer = GeminiAnalyzer()
        
        # Simple test query
        response = analyzer.client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Respond with 'Connection successful' if you receive this message."
        )
        
        return "successful" in response.text.lower() if response.text else False
        
    except Exception as e:
        print(f"❌ Gemini connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test connection
    if test_gemini_connection():
        print("✅ Gemini AI integration ready!")
    else:
        print("❌ Gemini AI connection failed")