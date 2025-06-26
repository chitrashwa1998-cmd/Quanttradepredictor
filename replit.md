# TribexAlpha - Trading Analytics Platform

## Overview

TribexAlpha is a comprehensive trading analytics platform built with Streamlit that provides volatility forecasting, model training, and backtesting capabilities for financial markets. The application focuses on technical analysis and machine learning-based predictions using OHLC (Open, High, Low, Close) price data.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **UI Components**: Multi-page application with cyberpunk-themed custom CSS styling
- **Pages**: Data Upload, Model Training, Predictions, Backtesting, Database Manager
- **Visualization**: Plotly for interactive charts and technical analysis displays

### Backend Architecture
- **Database Layer**: PostgreSQL-only database adapter with unified interface
- **Model Layer**: XGBoost, CatBoost, and Random Forest ensemble models
- **Feature Engineering**: Technical indicators using TA library
- **Data Processing**: Pandas-based OHLC data validation and cleaning

### Machine Learning Pipeline
- **Primary Model**: Volatility prediction (regression)
- **Feature Engineering**: Technical indicators, lagged features, custom volatility metrics
- **Model Training**: Ensemble approach with multiple algorithms
- **Prediction Engine**: Real-time volatility forecasting

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
- **Runtime**: Python 3.11
- **Port**: 5000 (mapped to port 80 externally)
- **Packages**: PostgreSQL, GCC, and OpenCL support via Nix

### Environment Setup
- PostgreSQL database creation required
- DATABASE_URL automatically configured by Replit
- Custom CSS and styling files served statically

### Session Management
- Streamlit session state for data persistence
- Auto-restore system for model recovery
- Database backup for long-term persistence

## Changelog
- June 26, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.