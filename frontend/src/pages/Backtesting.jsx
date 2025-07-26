/**
 * Backtesting page - Exact Streamlit UI replication
 */

import { useState, useEffect } from 'react';
import { dataAPI } from '../services/api';

const Backtesting = () => {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [loading, setLoading] = useState(false);
  const [backtestStatus, setBacktestStatus] = useState('');
  const [backtestResults, setBacktestResults] = useState(null);

  // Load datasets
  const loadDatasets = async () => {
    try {
      setLoading(true);
      const response = await dataAPI.getDatasets();
      const datasetList = Array.isArray(response?.data) ? response.data : [];
      setDatasets(datasetList);

      // Auto-select first dataset if available
      if (datasetList.length > 0) {
        setSelectedDataset(datasetList[0].name);
      }
    } catch (error) {
      console.error('Error loading datasets:', error);
      setBacktestStatus(`❌ Error loading datasets: ${error?.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  return (
    <div style={{ backgroundColor: '#0a0a0f', minHeight: '100vh', color: '#ffffff', fontFamily: 'Space Grotesk, sans-serif' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
        
        {/* Header */}
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem', fontFamily: 'Orbitron, monospace' }}>
          📈 Backtesting & Performance Analysis
        </h1>
        <p style={{ color: '#b8bcc8', fontSize: '1.1rem', marginBottom: '2rem' }}>
          Test and validate your trading strategies with historical data.
        </p>

        {/* Status */}
        {backtestStatus && (
          <div style={{
            padding: '1rem',
            borderRadius: '8px',
            marginBottom: '2rem',
            backgroundColor: backtestStatus.includes('✅') ? 'rgba(0, 255, 65, 0.1)' : 'rgba(255, 0, 128, 0.1)',
            border: `1px solid ${backtestStatus.includes('✅') ? 'rgba(0, 255, 65, 0.3)' : 'rgba(255, 0, 128, 0.3)'}`,
            color: backtestStatus.includes('✅') ? '#00ff41' : '#ff0080'
          }}>
            {backtestStatus}
          </div>
        )}

        {/* Dataset Selection */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>📊 Dataset Selection</h2>
          
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
              Select Dataset for Backtesting:
            </label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              style={{
                width: '100%',
                maxWidth: '400px',
                padding: '0.75rem',
                backgroundColor: 'rgba(25, 25, 45, 0.5)',
                border: '1px solid rgba(0, 255, 255, 0.3)',
                borderRadius: '4px',
                color: '#ffffff',
                fontSize: '1rem'
              }}
              disabled={loading}
            >
              <option value="">Choose dataset for backtesting</option>
              {datasets.map((dataset, index) => (
                <option key={index} value={dataset?.name || ''}>
                  {dataset?.name || 'Unknown'} ({dataset?.rows || 0} rows)
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Strategy Configuration */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>⚙️ Strategy Configuration</h2>
          
          <div style={{
            backgroundColor: 'rgba(25, 25, 45, 0.5)',
            border: '1px solid rgba(0, 255, 255, 0.3)',
            borderRadius: '8px',
            padding: '2rem'
          }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem' }}>
              <div>
                <h3 style={{ color: '#00ffff', marginBottom: '1rem' }}>📈 Entry Strategy</h3>
                <div style={{ marginBottom: '1rem' }}>
                  <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
                    Entry Signal:
                  </label>
                  <select style={{
                    width: '100%',
                    padding: '0.75rem',
                    backgroundColor: 'rgba(25, 25, 45, 0.5)',
                    border: '1px solid rgba(0, 255, 255, 0.3)',
                    borderRadius: '4px',
                    color: '#ffffff'
                  }}>
                    <option>Volatility Breakout</option>
                    <option>Direction Prediction</option>
                    <option>Reversal Detection</option>
                    <option>Combined Signals</option>
                  </select>
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
                    Confidence Threshold:
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="95"
                    defaultValue="70"
                    style={{ width: '100%', marginBottom: '0.5rem' }}
                  />
                  <div style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>70%</div>
                </div>
              </div>
              
              <div>
                <h3 style={{ color: '#ff0080', marginBottom: '1rem' }}>🛡️ Risk Management</h3>
                <div style={{ marginBottom: '1rem' }}>
                  <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
                    Stop Loss (%):
                  </label>
                  <input
                    type="number"
                    defaultValue="2"
                    min="0.5"
                    max="10"
                    step="0.5"
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      backgroundColor: 'rgba(25, 25, 45, 0.5)',
                      border: '1px solid rgba(0, 255, 255, 0.3)',
                      borderRadius: '4px',
                      color: '#ffffff'
                    }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
                    Take Profit (%):
                  </label>
                  <input
                    type="number"
                    defaultValue="4"
                    min="1"
                    max="20"
                    step="0.5"
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      backgroundColor: 'rgba(25, 25, 45, 0.5)',
                      border: '1px solid rgba(0, 255, 255, 0.3)',
                      borderRadius: '4px',
                      color: '#ffffff'
                    }}
                  />
                </div>
              </div>
              
              <div>
                <h3 style={{ color: '#ffd700', marginBottom: '1rem' }}>💰 Position Sizing</h3>
                <div style={{ marginBottom: '1rem' }}>
                  <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
                    Initial Capital:
                  </label>
                  <input
                    type="number"
                    defaultValue="100000"
                    min="10000"
                    max="10000000"
                    step="10000"
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      backgroundColor: 'rgba(25, 25, 45, 0.5)',
                      border: '1px solid rgba(0, 255, 255, 0.3)',
                      borderRadius: '4px',
                      color: '#ffffff'
                    }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
                    Risk per Trade (%):
                  </label>
                  <input
                    type="number"
                    defaultValue="1"
                    min="0.1"
                    max="5"
                    step="0.1"
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      backgroundColor: 'rgba(25, 25, 45, 0.5)',
                      border: '1px solid rgba(0, 255, 255, 0.3)',
                      borderRadius: '4px',
                      color: '#ffffff'
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Run Backtest Button */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <button
            style={{
              backgroundColor: '#00ff41',
              color: '#0a0a0f',
              border: 'none',
              padding: '1rem 2rem',
              borderRadius: '8px',
              fontSize: '1.1rem',
              fontWeight: 'bold',
              cursor: 'pointer',
              minWidth: '250px'
            }}
            disabled={loading || !selectedDataset}
          >
            {loading ? '📊 Running Backtest...' : '📊 Run Backtest'}
          </button>
          
          <div style={{ marginTop: '0.5rem', color: '#b8bcc8', fontSize: '0.9rem' }}>
            {!selectedDataset ? 'Please select a dataset first' : 'Ready to run backtest analysis'}
          </div>
        </div>

        {/* Placeholder Results */}
        <div style={{
          backgroundColor: 'rgba(25, 25, 45, 0.5)',
          border: '1px solid rgba(0, 255, 255, 0.3)',
          borderRadius: '8px',
          padding: '2rem',
          textAlign: 'center'
        }}>
          <h2 style={{ color: '#00ffff', marginBottom: '1rem' }}>📊 Backtest Results</h2>
          <div style={{ color: '#b8bcc8', fontSize: '1.1rem' }}>
            Configure your strategy and run a backtest to see performance metrics, trade analysis, and risk statistics here.
          </div>
          <div style={{ marginTop: '1rem', fontSize: '3rem' }}>📈</div>
        </div>
      </div>
    </div>
  );
};

export default Backtesting;