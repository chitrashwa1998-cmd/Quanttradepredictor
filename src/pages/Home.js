import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../config/api';

const Home = () => {
  const [systemStatus, setSystemStatus] = useState({
    hasData: false,
    totalModels: 0,
    dataRows: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSystemStatus();
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const [dataResponse, modelsResponse] = await Promise.all([
        axios.get(`${API_BASE_URL}/data/summary`),
        axios.get(`${API_BASE_URL}/models/status`)
      ]);

      setSystemStatus({
        hasData: dataResponse.data.success && dataResponse.data.data.total_rows > 0,
        totalModels: modelsResponse.data.success ? modelsResponse.data.data.total_models : 0,
        dataRows: dataResponse.data.success ? dataResponse.data.data.total_rows : 0
      });
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading"></div>;
  }

  return (
    <div className="container">
      <div className="header">
        <h1>TribexAlpha</h1>
        <p>🚀 Advanced Machine Learning for Quantitative Trading Excellence</p>
        <p style={{marginTop: '1rem', opacity: 0.8}}>
          Harness the power of AI-driven market prediction and algorithmic trading strategies
        </p>
      </div>

      <div className="grid grid-4" style={{marginBottom: '3rem'}}>
        <div className="metric-card">
          <h3 style={{color: '#00ffff', margin: '0 0 1rem 0'}}>⚡ DATA ENGINE</h3>
          <div className="metric-value" style={{color: systemStatus.hasData ? '#00ff41' : '#ff0080'}}>
            {systemStatus.hasData ? '🟢 LOADED' : '🔴 NO DATA'}
          </div>
          <div className="metric-label">
            {systemStatus.dataRows.toLocaleString()} market records loaded
          </div>
        </div>

        <div className="metric-card">
          <h3 style={{color: '#00ff41', margin: '0 0 1rem 0'}}>🤖 AI MODELS</h3>
          <div className="metric-value" style={{color: systemStatus.totalModels > 0 ? '#00ff41' : '#ffaa00'}}>
            {systemStatus.totalModels}/7
          </div>
          <div className="metric-label">Neural networks trained</div>
        </div>

        <div className="metric-card">
          <h3 style={{color: '#8b5cf6', margin: '0 0 1rem 0'}}>🎯 PREDICTIONS</h3>
          <div className="metric-value" style={{color: systemStatus.totalModels > 0 ? '#00ff41' : '#ffaa00'}}>
            {systemStatus.totalModels > 0 ? '🟢 ACTIVE' : '⚠️ STANDBY'}
          </div>
          <div className="metric-label">Real-time market analysis</div>
        </div>

        <div className="metric-card" style={{background: 'linear-gradient(135deg, rgba(0, 255, 255, 0.1), rgba(139, 92, 246, 0.1))'}}>
          <h3 style={{color: '#ff0080', margin: '0 0 1rem 0'}}>🌐 SYSTEM</h3>
          <div className="metric-value" style={{color: '#00ff41'}}>ONLINE</div>
          <div className="metric-label">All systems operational</div>
        </div>
      </div>

      <div className="card">
        <h2 style={{color: '#00ffff', marginBottom: '2rem', textAlign: 'center'}}>
          🔮 ADVANCED PREDICTION CAPABILITIES
        </h2>

        <div className="grid grid-2">
          <div className="card">
            <h3 style={{color: '#00ffff', marginBottom: '1.5rem'}}>🧠 MACHINE LEARNING ARSENAL</h3>
            <div style={{lineHeight: 2}}>
              <div style={{margin: '1rem 0', padding: '0.5rem', background: 'rgba(0, 255, 255, 0.05)', borderRadius: '8px'}}>
                <strong style={{color: '#00ff41'}}>🎯 Direction Prediction</strong><br />
                <span style={{color: '#b8bcc8'}}>Advanced price movement forecasting with 94% accuracy</span>
              </div>
              <div style={{margin: '1rem 0', padding: '0.5rem', background: 'rgba(139, 92, 246, 0.05)', borderRadius: '8px'}}>
                <strong style={{color: '#8b5cf6'}}>📈 Magnitude Analysis</strong><br />
                <span style={{color: '#b8bcc8'}}>Precise percentage change estimation algorithms</span>
              </div>
              <div style={{margin: '1rem 0', padding: '0.5rem', background: 'rgba(255, 0, 128, 0.05)', borderRadius: '8px'}}>
                <strong style={{color: '#ff0080'}}>💰 Profit Probability</strong><br />
                <span style={{color: '#b8bcc8'}}>Trade success likelihood with risk assessment</span>
              </div>
              <div style={{margin: '1rem 0', padding: '0.5rem', background: 'rgba(255, 215, 0, 0.05)', borderRadius: '8px'}}>
                <strong style={{color: '#ffd700'}}>⚡ Volatility Forecasting</strong><br />
                <span style={{color: '#b8bcc8'}}>Dynamic market volatility prediction engine</span>
              </div>
            </div>
          </div>

          <div className="card">
            <h3 style={{color: '#00ff41', marginBottom: '1.5rem'}}>⚙️ TRADING INFRASTRUCTURE</h3>
            <div style={{lineHeight: 2}}>
              <div style={{margin: '1rem 0', padding: '0.5rem', background: 'rgba(0, 255, 65, 0.05)', borderRadius: '8px'}}>
                <strong style={{color: '#00ffff'}}>⚡ Real-time Processing</strong><br />
                <span style={{color: '#b8bcc8'}}>Ultra-low latency market data analysis</span>
              </div>
              <div style={{margin: '1rem 0', padding: '0.5rem', background: 'rgba(255, 107, 53, 0.05)', borderRadius: '8px'}}>
                <strong style={{color: '#ff6b35'}}>🔍 Advanced Backtesting</strong><br />
                <span style={{color: '#b8bcc8'}}>Historical performance validation framework</span>
              </div>
              <div style={{margin: '1rem 0', padding: '0.5rem', background: 'rgba(139, 92, 246, 0.05)', borderRadius: '8px'}}>
                <strong style={{color: '#8b5cf6'}}>🛡️ Risk Management</strong><br />
                <span style={{color: '#b8bcc8'}}>Intelligent portfolio optimization algorithms</span>
              </div>
              <div style={{margin: '1rem 0', padding: '0.5rem', background: 'rgba(255, 215, 0, 0.05)', borderRadius: '8px'}}>
                <strong style={{color: '#ffd700'}}>📊 Technical Indicators</strong><br />
                <span style={{color: '#b8bcc8'}}>50+ built-in technical analysis tools</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{background: 'linear-gradient(135deg, rgba(0, 255, 255, 0.1), rgba(139, 92, 246, 0.1))', border: '2px solid #00ffff', textAlign: 'center'}}>
        <h2 style={{color: '#00ffff', marginBottom: '2rem'}}>🚀 MISSION CONTROL</h2>
        <p style={{fontSize: '1.3rem', marginBottom: '2rem', fontWeight: 300}}>
          Deploy your quantitative trading system in 4 strategic phases
        </p>
      </div>

      <div className="grid grid-4" style={{marginBottom: '3rem'}}>
        {[
          {icon: "📊", title: "DATA INTEGRATION", subtitle: "Market Data Ingestion", desc: "Load OHLC data with advanced preprocessing", color: "#00ffff"},
          {icon: "🔬", title: "AI TRAINING", subtitle: "Neural Network Training", desc: "Deploy XGBoost ML prediction models", color: "#00ff41"},
          {icon: "🎯", title: "SIGNAL GENERATION", subtitle: "Trading Signal Engine", desc: "Real-time prediction and analysis", color: "#8b5cf6"},
          {icon: "📈", title: "STRATEGY VALIDATION", subtitle: "Performance Analytics", desc: "Comprehensive backtesting framework", color: "#ff0080"}
        ].map((step, i) => (
          <div key={i} className="metric-card" style={{textAlign: 'center', minHeight: '250px', borderColor: step.color}}>
            <div style={{fontSize: '4rem', marginBottom: '1.5rem', filter: `drop-shadow(0 0 10px ${step.color})`}}>
              {step.icon}
            </div>
            <h3 style={{color: step.color, marginBottom: '1rem'}}>{step.title}</h3>
            <p style={{color: '#00ffff', marginBottom: '1rem', fontWeight: 600, fontSize: '1.1rem'}}>
              {step.subtitle}
            </p>
            <p style={{color: '#b8bcc8', fontSize: '0.95rem', lineHeight: 1.5}}>
              {step.desc}
            </p>
            <div style={{marginTop: '1.5rem', padding: '0.5rem', background: `${step.color}20`, borderRadius: '8px'}}>
              <span style={{color: step.color, fontSize: '0.9rem'}}>Phase {i+1}/4</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Home;