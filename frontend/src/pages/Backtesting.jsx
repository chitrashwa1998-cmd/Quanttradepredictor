/**
 * Backtesting page - Complete Streamlit functionality migration
 */

import { useState, useEffect, useCallback } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI, predictionsAPI } from '../services/api';

const Backtesting = () => {
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState({});
  const [currentData, setCurrentData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [backtestConfig, setBacktestConfig] = useState({
    initial_capital: 10000,
    commission: 0.001,
    selected_model: '',
    strategy_type: 'Simple Direction',
    confidence_threshold: 0.7,
    prob_threshold: 0.7,
    stop_loss_pct: 5,
    take_profit_pct: 10,
    backtest_period: 'Last year'
  });
  const [backtestResults, setBacktestResults] = useState(null);
  const [backtestStatus, setBacktestStatus] = useState('');

  // Load initial data
  const loadInitialData = useCallback(async () => {
    try {
      setLoading(true);
      
      const [datasetsResponse, modelsResponse] = await Promise.all([
        dataAPI.getDatasets(),
        predictionsAPI.getModelsStatus()
      ]);

      setDatasets(datasetsResponse.data || []);
      const modelStatus = modelsResponse.data || {};
      setModels(modelStatus);

      // Auto-select first available trading model
      const tradingModels = Object.keys(modelStatus).filter(name => 
        modelStatus[name]?.loaded && ['direction', 'profit_probability'].includes(name)
      );

      if (tradingModels.length > 0) {
        setBacktestConfig(prev => ({
          ...prev,
          selected_model: tradingModels[0]
        }));
      }

      // Load data from preferred dataset
      const datasetList = datasetsResponse.data || [];
      if (datasetList.length > 0) {
        const preferredDatasets = ['training_dataset', 'main_dataset'];
        let datasetToLoad = null;

        for (const preferred of preferredDatasets) {
          const found = datasetList.find(d => d.name === preferred);
          if (found) {
            datasetToLoad = preferred;
            break;
          }
        }

        if (!datasetToLoad && datasetList.length > 0) {
          datasetToLoad = datasetList[0].name;
        }

        if (datasetToLoad) {
          const dataResponse = await dataAPI.loadDataset(datasetToLoad);
          setCurrentData(dataResponse.data);
          setBacktestStatus(`✅ Loaded data: ${dataResponse.data?.length || 0} records from ${datasetToLoad}`);
        }
      }

    } catch (error) {
      setBacktestStatus(`❌ Error loading data: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  // Run backtest
  const runBacktest = async () => {
    if (!currentData || currentData.length === 0) {
      setBacktestStatus('❌ No data loaded for backtesting');
      return;
    }

    if (!backtestConfig.selected_model) {
      setBacktestStatus('❌ Please select a trading model');
      return;
    }

    if (!models[backtestConfig.selected_model]?.loaded) {
      setBacktestStatus(`❌ ${backtestConfig.selected_model} model not trained`);
      return;
    }

    try {
      setLoading(true);
      setBacktestStatus('🚀 Running backtest...');

      const response = await predictionsAPI.runBacktest({
        model_type: backtestConfig.selected_model,
        config: backtestConfig,
        data: currentData
      });

      setBacktestResults(response.data);
      setBacktestStatus('✅ Backtest completed successfully!');
    } catch (error) {
      setBacktestStatus(`❌ Backtest failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const tradingModels = Object.keys(models).filter(name => 
    models[name]?.loaded && ['direction', 'profit_probability'].includes(name)
  );

  const strategyTypes = {
    'direction': ['Simple Direction', 'Direction with Confidence', 'Direction with Stop Loss'],
    'profit_probability': ['Profit Probability']
  };

  return (
    <div className="container mx-auto px-6 py-8">
      {/* Header */}
      <div className="trading-header mb-8">
        <h1 style={{
          margin: '0',
          background: 'var(--gradient-primary)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          fontFamily: 'var(--font-display)',
          fontSize: '2.5rem'
        }}>
          📈 BACKTEST ENGINE
        </h1>
        <p style={{
          fontSize: '1.2rem',
          margin: '1rem 0 0 0',
          color: 'rgba(255,255,255,0.8)',
          fontFamily: 'var(--font-primary)'
        }}>
          Strategy Performance Analysis
        </p>
      </div>

      {/* Data & Model Check */}
      {(!currentData || tradingModels.length === 0) && (
        <Card style={{ marginBottom: '2rem', background: 'rgba(255, 165, 0, 0.05)', border: '1px solid rgba(255, 165, 0, 0.2)' }}>
          <h3 style={{ color: '#ffa500', margin: '0 0 1rem 0' }}>
            ⚠️ Prerequisites Required
          </h3>
          {!currentData && (
            <p style={{ color: 'var(--text-primary)', margin: '0 0 0.5rem 0' }}>
              • No data loaded. Please go to the Data Upload page first.
            </p>
          )}
          {tradingModels.length === 0 && (
            <p style={{ color: 'var(--text-primary)', margin: '0' }}>
              • No trained models found. Please train Direction or Profit Probability models first.
            </p>
          )}
        </Card>
      )}

      {/* Backtesting Configuration */}
      <Card style={{ marginBottom: '2rem' }}>
        <h2 style={{
          color: 'var(--accent-cyan)',
          fontFamily: 'var(--font-display)',
          fontSize: '1.5rem',
          marginBottom: '1.5rem'
        }}>
          ⚙️ Backtesting Configuration
        </h2>

        {/* Basic Parameters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label style={{
              display: 'block',
              color: 'var(--text-primary)',
              marginBottom: '0.5rem',
              fontWeight: '500'
            }}>
              Initial Capital ($):
            </label>
            <input
              type="number"
              min="1000"
              max="1000000"
              step="1000"
              value={backtestConfig.initial_capital}
              onChange={(e) => setBacktestConfig(prev => ({
                ...prev,
                initial_capital: parseInt(e.target.value)
              }))}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'var(--bg-secondary)',
                border: '2px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-primary)'
              }}
            />
          </div>

          <div>
            <label style={{
              display: 'block',
              color: 'var(--text-primary)',
              marginBottom: '0.5rem',
              fontWeight: '500'
            }}>
              Commission Rate (%):
            </label>
            <input
              type="number"
              min="0"
              max="5"
              step="0.01"
              value={backtestConfig.commission * 100}
              onChange={(e) => setBacktestConfig(prev => ({
                ...prev,
                commission: parseFloat(e.target.value) / 100
              }))}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'var(--bg-secondary)',
                border: '2px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-primary)'
              }}
            />
          </div>

          <div>
            <label style={{
              display: 'block',
              color: 'var(--text-primary)',
              marginBottom: '0.5rem',
              fontWeight: '500'
            }}>
              Trading Model:
            </label>
            <select
              value={backtestConfig.selected_model}
              onChange={(e) => setBacktestConfig(prev => ({
                ...prev,
                selected_model: e.target.value
              }))}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'var(--bg-secondary)',
                border: '2px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-primary)'
              }}
            >
              <option value="">Select model...</option>
              {tradingModels.map(model => (
                <option key={model} value={model}>
                  {model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Strategy Configuration */}
        <div style={{ marginBottom: '1.5rem' }}>
          <h3 style={{
            color: 'var(--accent-gold)',
            fontFamily: 'var(--font-display)',
            fontSize: '1.2rem',
            marginBottom: '1rem'
          }}>
            📊 Strategy Configuration
          </h3>

          {backtestConfig.selected_model === 'direction' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label style={{
                  display: 'block',
                  color: 'var(--text-primary)',
                  marginBottom: '0.5rem',
                  fontWeight: '500'
                }}>
                  Strategy Type:
                </label>
                <select
                  value={backtestConfig.strategy_type}
                  onChange={(e) => setBacktestConfig(prev => ({
                    ...prev,
                    strategy_type: e.target.value
                  }))}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'var(--bg-secondary)',
                    border: '2px solid var(--border)',
                    borderRadius: '8px',
                    color: 'var(--text-primary)',
                    fontFamily: 'var(--font-primary)'
                  }}
                >
                  {strategyTypes.direction.map(strategy => (
                    <option key={strategy} value={strategy}>{strategy}</option>
                  ))}
                </select>
              </div>

              {backtestConfig.strategy_type === 'Direction with Confidence' && (
                <div>
                  <label style={{
                    display: 'block',
                    color: 'var(--text-primary)',
                    marginBottom: '0.5rem',
                    fontWeight: '500'
                  }}>
                    Confidence Threshold: {(backtestConfig.confidence_threshold * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="0.95"
                    step="0.05"
                    value={backtestConfig.confidence_threshold}
                    onChange={(e) => setBacktestConfig(prev => ({
                      ...prev,
                      confidence_threshold: parseFloat(e.target.value)
                    }))}
                    style={{
                      width: '100%',
                      accentColor: 'var(--accent-cyan)'
                    }}
                  />
                </div>
              )}

              {backtestConfig.strategy_type === 'Direction with Stop Loss' && (
                <>
                  <div>
                    <label style={{
                      display: 'block',
                      color: 'var(--text-primary)',
                      marginBottom: '0.5rem',
                      fontWeight: '500'
                    }}>
                      Stop Loss: {backtestConfig.stop_loss_pct}%
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      step="1"
                      value={backtestConfig.stop_loss_pct}
                      onChange={(e) => setBacktestConfig(prev => ({
                        ...prev,
                        stop_loss_pct: parseInt(e.target.value)
                      }))}
                      style={{
                        width: '100%',
                        accentColor: 'var(--accent-cyan)'
                      }}
                    />
                  </div>
                  <div>
                    <label style={{
                      display: 'block',
                      color: 'var(--text-primary)',
                      marginBottom: '0.5rem',
                      fontWeight: '500'
                    }}>
                      Take Profit: {backtestConfig.take_profit_pct}%
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      step="1"
                      value={backtestConfig.take_profit_pct}
                      onChange={(e) => setBacktestConfig(prev => ({
                        ...prev,
                        take_profit_pct: parseInt(e.target.value)
                      }))}
                      style={{
                        width: '100%',
                        accentColor: 'var(--accent-cyan)'
                      }}
                    />
                  </div>
                </>
              )}
            </div>
          )}

          {backtestConfig.selected_model === 'profit_probability' && (
            <div>
              <label style={{
                display: 'block',
                color: 'var(--text-primary)',
                marginBottom: '0.5rem',
                fontWeight: '500'
              }}>
                Minimum Profit Probability: {(backtestConfig.prob_threshold * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.5"
                max="0.95"
                step="0.05"
                value={backtestConfig.prob_threshold}
                onChange={(e) => setBacktestConfig(prev => ({
                  ...prev,
                  prob_threshold: parseFloat(e.target.value)
                }))}
                style={{
                  width: '100%',
                  accentColor: 'var(--accent-cyan)',
                  maxWidth: '300px'
                }}
              />
            </div>
          )}
        </div>

        {/* Backtesting Period */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label style={{
              display: 'block',
              color: 'var(--text-primary)',
              marginBottom: '0.5rem',
              fontWeight: '500'
            }}>
              Backtesting Period:
            </label>
            <select
              value={backtestConfig.backtest_period}
              onChange={(e) => setBacktestConfig(prev => ({
                ...prev,
                backtest_period: e.target.value
              }))}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'var(--bg-secondary)',
                border: '2px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-primary)'
              }}
            >
              <option value="Last 6 months">Last 6 months</option>
              <option value="Last year">Last year</option>
              <option value="Last 2 years">Last 2 years</option>
              <option value="All data">All data</option>
            </select>
          </div>
          <div style={{
            background: 'rgba(0, 255, 255, 0.05)',
            border: '1px solid rgba(0, 255, 255, 0.2)',
            borderRadius: '8px',
            padding: '1rem',
            alignSelf: 'end'
          }}>
            <p style={{ color: 'var(--accent-cyan)', margin: '0', fontSize: '0.9rem' }}>
              📊 Data available: {currentData?.length || 0} records
              <br />
              Period: {backtestConfig.backtest_period}
            </p>
          </div>
        </div>
      </Card>

      {/* Run Backtest */}
      <Card style={{ marginBottom: '2rem' }}>
        <div style={{ textAlign: 'center' }}>
          <button
            onClick={runBacktest}
            disabled={loading || !currentData || tradingModels.length === 0 || !backtestConfig.selected_model}
            style={{
              padding: '1rem 3rem',
              background: loading || !currentData || tradingModels.length === 0 || !backtestConfig.selected_model
                ? 'var(--bg-secondary)' 
                : 'var(--gradient-primary)',
              border: '2px solid var(--border-hover)',
              borderRadius: '12px',
              color: 'white',
              fontFamily: 'var(--font-primary)',
              fontWeight: '700',
              fontSize: '1.2rem',
              cursor: loading || !currentData || tradingModels.length === 0 || !backtestConfig.selected_model ? 'not-allowed' : 'pointer',
              transition: 'all 0.3s ease'
            }}
          >
            {loading ? '⏳ Running Backtest...' : '🚀 Run Backtest'}
          </button>
        </div>
      </Card>

      {/* Backtest Results */}
      {backtestResults && (
        <Card>
          <h2 style={{
            color: 'var(--accent-gold)',
            fontFamily: 'var(--font-display)',
            fontSize: '1.5rem',
            marginBottom: '1.5rem'
          }}>
            📊 Backtest Results
          </h2>

          {/* Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {backtestResults.metrics && Object.entries(backtestResults.metrics).map(([key, value]) => (
              <div key={key} style={{
                background: 'rgba(0, 255, 255, 0.05)',
                border: '1px solid rgba(0, 255, 255, 0.2)',
                borderRadius: '12px',
                padding: '1.5rem',
                textAlign: 'center'
              }}>
                <div style={{
                  color: 'var(--accent-gold)',
                  fontSize: '1.8rem',
                  fontWeight: '700',
                  marginBottom: '0.5rem',
                  fontFamily: 'var(--font-mono)'
                }}>
                  {typeof value === 'number' ? 
                    (key.includes('return') || key.includes('ratio') ? value.toFixed(2) : 
                     key.includes('rate') ? (value * 100).toFixed(1) + '%' : 
                     value.toLocaleString()) : 
                    value
                  }
                </div>
                <div style={{
                  color: 'var(--text-secondary)',
                  fontSize: '0.9rem',
                  textTransform: 'capitalize',
                  fontFamily: 'var(--font-primary)'
                }}>
                  {key.replace(/_/g, ' ')}
                </div>
              </div>
            ))}
          </div>

          {/* Trade History Table */}
          {backtestResults.trades && (
            <div>
              <h3 style={{
                color: 'var(--accent-cyan)',
                marginBottom: '1rem',
                fontFamily: 'var(--font-display)'
              }}>
                📈 Trade History (Last 10 trades)
              </h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.9rem'
                }}>
                  <thead>
                    <tr style={{ background: 'rgba(0, 255, 255, 0.1)' }}>
                      <th style={{ padding: '0.75rem', textAlign: 'left', color: 'var(--accent-cyan)' }}>Date</th>
                      <th style={{ padding: '0.75rem', textAlign: 'left', color: 'var(--accent-cyan)' }}>Action</th>
                      <th style={{ padding: '0.75rem', textAlign: 'left', color: 'var(--accent-cyan)' }}>Price</th>
                      <th style={{ padding: '0.75rem', textAlign: 'left', color: 'var(--accent-cyan)' }}>Quantity</th>
                      <th style={{ padding: '0.75rem', textAlign: 'left', color: 'var(--accent-cyan)' }}>P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {backtestResults.trades.slice(-10).map((trade, index) => (
                      <tr key={index}>
                        <td style={{ padding: '0.75rem', color: 'var(--text-primary)' }}>
                          {trade.date}
                        </td>
                        <td style={{ 
                          padding: '0.75rem', 
                          color: trade.action === 'BUY' ? '#51cf66' : '#ff6b6b',
                          fontWeight: '600'
                        }}>
                          {trade.action}
                        </td>
                        <td style={{ padding: '0.75rem', color: 'var(--text-primary)' }}>
                          ${trade.price?.toFixed(2)}
                        </td>
                        <td style={{ padding: '0.75rem', color: 'var(--text-primary)' }}>
                          {trade.quantity}
                        </td>
                        <td style={{ 
                          padding: '0.75rem', 
                          color: (trade.pnl || 0) >= 0 ? '#51cf66' : '#ff6b6b',
                          fontWeight: '600'
                        }}>
                          ${trade.pnl?.toFixed(2) || '0.00'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Status Message */}
      {backtestStatus && (
        <div style={{
          position: 'fixed',
          bottom: '2rem',
          right: '2rem',
          padding: '1rem 1.5rem',
          background: backtestStatus.includes('❌') 
            ? 'rgba(255, 0, 0, 0.9)' 
            : backtestStatus.includes('✅')
            ? 'rgba(0, 255, 0, 0.9)'
            : 'rgba(0, 255, 255, 0.9)',
          border: `1px solid ${
            backtestStatus.includes('❌') ? '#ff0000' : 
            backtestStatus.includes('✅') ? '#00ff00' : '#00ffff'
          }`,
          borderRadius: '8px',
          color: 'white',
          fontFamily: 'var(--font-primary)',
          fontWeight: '600',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
          zIndex: 1000,
          maxWidth: '400px'
        }}>
          {backtestStatus}
        </div>
      )}
    </div>
  );
};

export default Backtesting;