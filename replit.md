# TribexAlpha - Trading Analytics Platform

## Overview

TribexAlpha is a comprehensive trading analytics platform built with Streamlit that provides volatility forecasting, model training, and backtesting capabilities for financial markets. The application focuses on technical analysis and machine learning-based predictions using OHLC (Open, High, Low, Close) price data.

## System Architecture

### Frontend Architecture (MIGRATED TO REACT)
- **Framework**: React 18 with Vite build system (migrated from Streamlit)
- **UI Library**: Tailwind CSS with cyberpunk theme matching original design
- **Routing**: React Router for single-page application navigation
- **Pages**: Dashboard, DataUpload, ModelTraining, Predictions, LiveTrading, Backtesting, DatabaseManager
- **State Management**: React hooks with error boundaries
- **API Communication**: Axios for HTTP requests and WebSocket connections

### Backend Architecture (MIGRATED TO FASTAPI)
- **Framework**: FastAPI with async/await support (migrated from Streamlit)
- **API Structure**: RESTful endpoints with OpenAPI documentation
- **WebSocket Support**: Real-time market data and live predictions
- **Database Layer**: PostgreSQL via existing database adapter
- **Model Layer**: XGBoost, CatBoost, and Random Forest ensemble models
- **Feature Engineering**: Technical indicators using TA library
- **Data Processing**: Pandas-based OHLC data validation and cleaning

### Machine Learning Pipeline
- **Primary Model**: Volatility prediction (regression)
- **Secondary Models**: Reversal detection (classification) with comprehensive feature integration
- **Feature Engineering**: Technical indicators, lagged features, custom metrics, and time context features
- **Model Training**: Ensemble approach with multiple algorithms
- **Prediction Engine**: Real-time volatility forecasting and reversal detection

## Key Components

### Database Management (`utils/database_adapter.py`)
- PostgreSQL-exclusive database interface
- Unified adapter pattern for data persistence
- Automatic connection management and error handling
- Support for OHLC data, model results, and predictions storage

### Model Training (`models/`)
- **VolatilityModel**: Primary regression model for volatility prediction
- **ModelManager**: Centralized model management and persistence
- **XGBoostModels**: Legacy compatibility layer, now focused on volatility only
- **Feature Engineering**: Comprehensive technical indicator calculation

### Technical Indicators (`features/technical_indicators.py`)
- ATR (Average True Range)
- Bollinger Band Width
- Keltner Channel Width
- RSI (Relative Strength Index)
- Donchian Channel Width
- Custom volatility metrics and lagged features

### Auto-Restore System (`auto_restore.py`)
- Automatic model and data recovery on application restart
- Session state management
- Database persistence integration

### Gemini AI Integration (`utils/gemini_analysis.py`)
- Natural language market analysis using Google's Gemini 2.5 models
- Market sentiment analysis and confidence scoring
- AI-powered trading insights and recommendations
- Risk assessment with natural language explanations
- Integration with both static and live prediction displays

## Data Flow

1. **Data Upload**: CSV files uploaded through Streamlit interface
2. **Data Validation**: OHLC format validation and quality checks
3. **Feature Engineering**: Technical indicators calculation
4. **Model Training**: Volatility model training with ensemble methods
5. **Prediction Generation**: Real-time volatility forecasting
6. **Backtesting**: Strategy performance analysis
7. **Database Persistence**: All data, models, and results stored in PostgreSQL

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualization
- **XGBoost**: Gradient boosting framework
- **CatBoost**: Gradient boosting library
- **Scikit-learn**: Machine learning utilities

### Database
- **PostgreSQL**: Primary database (psycopg2-binary for connectivity)
- **DATABASE_URL**: Environment variable for database connection

### Technical Analysis
- **TA**: Technical analysis library
- **Pandas-TA**: Additional technical indicators
- **FINTA**: Financial technical analysis

### Data Sources
- **yfinance**: Yahoo Finance data integration (optional)

## Deployment Strategy

### Replit Configuration
- **Platform**: Replit autoscale deployment
- **Runtime**: Python 3.11 + Node.js 20
- **Services**: 
  - Frontend: React dev server on port 5000
  - Backend: FastAPI server on port 8000 (in development)
- **Packages**: PostgreSQL, GCC, and OpenCL support via Nix

### Environment Setup
- PostgreSQL database creation required
- DATABASE_URL automatically configured by Replit
- Frontend builds static assets with Vite
- Backend serves API endpoints with CORS support

### Session Management
- React state management with local storage persistence
- FastAPI database session handling
- Auto-restore system for model recovery
- Database backup for long-term persistence

## Changelog
- July 26, 2025: **MIGRATION COMPLETED** - Full Streamlit to React + FastAPI migration successfully completed
  - ✅ Created complete React frontend with Tailwind CSS cyberpunk theme
  - ✅ Built FastAPI backend with comprehensive RESTful API endpoints
  - ✅ Implemented React pages: Dashboard, DataUpload, ModelTraining, Predictions, LiveTrading, Backtesting, DatabaseManager
  - ✅ Added WebSocket support for real-time data streaming
  - ✅ Preserved all existing PostgreSQL database functionality with 4050+ records across 2 datasets
  - ✅ Maintained ML model integration (volatility, direction, profit probability, reversal) with all 4 models trained
  - ✅ Fixed API connectivity issues and added missing endpoints for complete functionality
  - ✅ Verified data upload works correctly (tested with livenifty50.csv - 225 rows)
  - ✅ All original Streamlit functionality preserved in modern React interface
- July 22, 2025: Integrated Google Gemini AI for advanced market analysis - added natural language insights, sentiment analysis, and AI-powered trading recommendations to both static and live predictions
- July 18, 2025: Fixed model persistence issue - models now automatically saved to database after training and persist across app restarts
- July 18, 2025: Fixed profit probability model RangeIndex error by preserving datetime indices throughout feature calculation
- July 17, 2025: Simplified candle detection to trigger predictions immediately when new 5-minute candles close (removed forced updates and buffers)
- July 17, 2025: Fixed predictions not updating after 5-minute candle close - improved candle detection logic and added forced prediction updates
- July 17, 2025: Fixed reconnection issue where predictions wouldn't restart after disconnect/reconnect cycle
- June 26, 2025: Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.