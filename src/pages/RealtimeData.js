
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { API_BASE_URL } from '../config/api';

const RealtimeData = () => {
  const [marketData, setMarketData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(false);

  useEffect(() => {
    fetchMarketData();
  }, []);

  useEffect(() => {
    let interval;
    if (autoRefresh) {
      interval = setInterval(fetchMarketData, 30000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const fetchMarketData = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/realtime/nifty`);
      setMarketData(response.data);
    } catch (error) {
      console.error('Failed to fetch market data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading"></div>;
  }

  if (!marketData) {
    return (
      <div className="container">
        <div className="alert alert-error">
          ❌ Could not fetch NIFTY 50 data. Market might be closed or there are connectivity issues.
        </div>
      </div>
    );
  }

  // Format dates to IST for chart
  const formatToIST = (dateStr) => {
    try {
      const date = new Date(dateStr);
      // Return the date formatted properly for IST display
      return date.toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (e) {
      return dateStr;
    }
  };

  const candlestickData = {
    x: marketData.chart_data.map(d => formatToIST(d.date)),
    open: marketData.chart_data.map(d => d.open),
    high: marketData.chart_data.map(d => d.high),
    low: marketData.chart_data.map(d => d.low),
    close: marketData.chart_data.map(d => d.close),
    type: 'candlestick',
    name: 'NIFTY 50'
  };

  return (
    <div className="container">
      <div className="header">
        <h1>📊 NIFTY 50 REALTIME ANALYSIS</h1>
        <p>Live Market Data with ML Predictions</p>
      </div>

      <div className="grid grid-3" style={{marginBottom: '2rem'}}>
        <div className="metric-card">
          <h3 style={{color: '#00ffff'}}>Market Status</h3>
          <div className="metric-value" style={{color: marketData.market_open ? '#00ff41' : '#ff0080'}}>
            {marketData.market_open ? '🟢 OPEN' : '🔴 CLOSED'}
          </div>
        </div>

        <div className="metric-card">
          <h3 style={{color: '#00ffff'}}>Current Time</h3>
          <div className="metric-value" style={{color: '#ffd700'}}>
            {new Date().toLocaleTimeString('en-IN', {timeZone: 'Asia/Kolkata'})}
          </div>
        </div>

        <div className="metric-card">
          <h3 style={{color: '#00ffff'}}>Trading Day</h3>
          <div className="metric-value" style={{color: '#ffd700'}}>
            {new Date().toLocaleDateString('en-IN')}
          </div>
        </div>
      </div>

      <div className="card">
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem'}}>
          <h3>📊 NIFTY 50 - 5 Minute Data</h3>
          <label style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto Refresh (30s)
          </label>
        </div>

        <div className="grid grid-4" style={{marginBottom: '2rem'}}>
          <div className="metric-card">
            <div className="metric-label">Current Price</div>
            <div className="metric-value">₹{marketData.current_price.toFixed(2)}</div>
            <div style={{color: marketData.price_change >= 0 ? '#00ff41' : '#ff0080'}}>
              {marketData.price_change >= 0 ? '+' : ''}{marketData.price_change.toFixed(2)} ({marketData.price_change_pct.toFixed(2)}%)
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-label">High</div>
            <div className="metric-value">₹{marketData.high.toFixed(2)}</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Low</div>
            <div className="metric-value">₹{marketData.low.toFixed(2)}</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Volume</div>
            <div className="metric-value">{marketData.volume.toLocaleString()}</div>
          </div>
        </div>

        <Plot
          data={[candlestickData]}
          layout={{
            title: `NIFTY 50 - 5m Candlestick Chart (Last Updated: ${new Date(marketData.last_updated).toLocaleTimeString()})`,
            xaxis: { title: 'Time', color: '#ffffff' },
            yaxis: { title: 'Price (₹)', color: '#ffffff' },
            height: 500,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#ffffff' }
          }}
          style={{width: '100%'}}
          config={{responsive: true}}
        />
      </div>

      <div className="card">
        <h3 style={{marginBottom: '1.5rem'}}>📈 Technical Analysis</h3>
        
        <div className="grid grid-3">
          {marketData.technical_indicators.rsi && (
            <div className="metric-card">
              <div className="metric-label">RSI</div>
              <div className="metric-value">{marketData.technical_indicators.rsi.toFixed(2)}</div>
              <div className="alert" style={{
                background: marketData.technical_indicators.rsi > 70 ? 'rgba(255, 0, 128, 0.1)' : 
                           marketData.technical_indicators.rsi < 30 ? 'rgba(255, 0, 128, 0.1)' : 
                           'rgba(0, 255, 255, 0.1)',
                color: marketData.technical_indicators.rsi > 70 ? '#ff0080' : 
                       marketData.technical_indicators.rsi < 30 ? '#ff0080' : '#00ffff'
              }}>
                {marketData.technical_indicators.rsi > 70 ? 'Overbought' : 
                 marketData.technical_indicators.rsi < 30 ? 'Oversold' : 'Neutral'}
              </div>
            </div>
          )}

          {marketData.technical_indicators.macd && (
            <div className="metric-card">
              <div className="metric-label">MACD Histogram</div>
              <div className="metric-value">{marketData.technical_indicators.macd.toFixed(4)}</div>
              <div className="alert" style={{
                background: marketData.technical_indicators.macd > 0 ? 'rgba(0, 255, 65, 0.1)' : 'rgba(255, 0, 128, 0.1)',
                color: marketData.technical_indicators.macd > 0 ? '#00ff41' : '#ff0080'
              }}>
                {marketData.technical_indicators.macd > 0 ? 'Bullish' : 'Bearish'}
              </div>
            </div>
          )}

          {marketData.technical_indicators.bb_position && (
            <div className="metric-card">
              <div className="metric-label">Bollinger Position</div>
              <div className="metric-value">{marketData.technical_indicators.bb_position.toFixed(2)}</div>
              <div className="alert" style={{
                background: marketData.technical_indicators.bb_position > 0.8 ? 'rgba(255, 0, 128, 0.1)' : 
                           marketData.technical_indicators.bb_position < 0.2 ? 'rgba(255, 0, 128, 0.1)' : 
                           'rgba(0, 255, 255, 0.1)',
                color: marketData.technical_indicators.bb_position > 0.8 ? '#ff0080' : 
                       marketData.technical_indicators.bb_position < 0.2 ? '#ff0080' : '#00ffff'
              }}>
                {marketData.technical_indicators.bb_position > 0.8 ? 'Overbought' : 
                 marketData.technical_indicators.bb_position < 0.2 ? 'Oversold' : 'Normal'}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <h3 style={{marginBottom: '1.5rem'}}>🤖 Machine Learning Predictions</h3>
        <div className="alert alert-warning">
          ⚠️ ML prediction system integration in progress. Train models first in the Model Training page.
        </div>
      </div>

      <div className="card">
        <h3 style={{marginBottom: '1.5rem'}}>📁 Data Management</h3>
        
        <div className="grid grid-3">
          <button className="btn btn-secondary" onClick={() => {
            const csv = marketData.chart_data.map(d => 
              `${d.date},${d.open},${d.high},${d.low},${d.close},${d.volume}`
            ).join('\n');
            const csvWithHeader = 'Date,Open,High,Low,Close,Volume\n' + csv;
            const blob = new Blob([csvWithHeader], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'nifty50_5m_data.csv';
            a.click();
          }}>
            📥 Download CSV
          </button>

          <button className="btn btn-secondary" onClick={() => {
            // Save to database functionality would go here
            alert('Save to database functionality will be implemented');
          }}>
            💾 Save to Database
          </button>

          <button className="btn btn-primary" onClick={fetchMarketData}>
            🔄 Refresh Data
          </button>
        </div>
      </div>
    </div>
  );
};

export default RealtimeData;
