/**
 * Live Data page - Complete Streamlit functionality migration
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { liveDataAPI } from '../services/api';

const LiveData = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [liveData, setLiveData] = useState({});
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const [loading, setLoading] = useState(false);
  const [credentials, setCredentials] = useState({
    access_token: '',
    api_key: ''
  });
  const [selectedInstruments, setSelectedInstruments] = useState(['NIFTY 50', 'BANK NIFTY']);
  const [customInstrument, setCustomInstrument] = useState('');
  const [predictions, setPredictions] = useState({});
  const [isPredictionPipelineActive, setIsPredictionPipelineActive] = useState(false);
  const [status, setStatus] = useState('');
  const [activeTab, setActiveTab] = useState('config');
  const wsRef = useRef(null);

  // Instrument mappings for Indian market
  const popularInstruments = {
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "BANK NIFTY": "NSE_INDEX|Nifty Bank",
    "NIFTY IT": "NSE_INDEX|Nifty IT",
    "NIFTY FMCG": "NSE_INDEX|Nifty FMCG",
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "HDFC BANK": "NSE_EQ|INE040A01034",
    "INFOSYS": "NSE_EQ|INE009A01021"
  };

  // Connect to live data feed
  const connectToLiveData = async () => {
    if (!credentials.access_token || !credentials.api_key) {
      setStatus('❌ Please provide both Access Token and API Key');
      return;
    }

    try {
      setLoading(true);
      setStatus('🔌 Connecting to Upstox WebSocket...');

      const instruments = selectedInstruments.map(name => 
        popularInstruments[name] || name
      );

      if (customInstrument) {
        instruments.push(customInstrument);
      }

      const response = await liveDataAPI.connect({
        access_token: credentials.access_token,
        api_key: credentials.api_key,
        instruments
      });

      if (response.data.success) {
        setIsConnected(true);
        setConnectionStatus('Connected');
        setStatus('✅ Connected to live data feed successfully!');
        
        // Start WebSocket connection for real-time updates
        startWebSocketConnection();
      } else {
        setStatus(`❌ Connection failed: ${response.data.message}`);
      }
    } catch (error) {
      setStatus(`❌ Connection error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Start WebSocket connection
  const startWebSocketConnection = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/live-data`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('Connected (Live)');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'market_data') {
          setLiveData(prev => ({
            ...prev,
            [data.instrument]: data.data
          }));
        } else if (data.type === 'prediction') {
          setPredictions(prev => ({
            ...prev,
            [data.instrument]: data.prediction
          }));
        }
      } catch (error) {
        console.error('WebSocket data parsing error:', error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('Disconnected');
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setStatus('❌ WebSocket connection error');
    };
  };

  // Disconnect from live data
  const disconnectFromLiveData = async () => {
    try {
      setLoading(true);
      setStatus('🔌 Disconnecting from live data...');

      if (wsRef.current) {
        wsRef.current.close();
      }

      await liveDataAPI.disconnect();
      
      setIsConnected(false);
      setConnectionStatus('Disconnected');
      setLiveData({});
      setPredictions({});
      setStatus('✅ Disconnected from live data feed');
    } catch (error) {
      setStatus(`❌ Disconnect error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Start prediction pipeline
  const startPredictionPipeline = async () => {
    if (!isConnected) {
      setStatus('❌ Please connect to live data first');
      return;
    }

    try {
      setLoading(true);
      setStatus('🚀 Starting live prediction pipeline...');

      const response = await liveDataAPI.startPredictionPipeline();
      
      if (response.data.success) {
        setIsPredictionPipelineActive(true);
        setStatus('✅ Live prediction pipeline started successfully!');
      } else {
        setStatus(`❌ Failed to start prediction pipeline: ${response.data.message}`);
      }
    } catch (error) {
      setStatus(`❌ Pipeline error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Stop prediction pipeline
  const stopPredictionPipeline = async () => {
    try {
      setLoading(true);
      setStatus('🛑 Stopping prediction pipeline...');

      await liveDataAPI.stopPredictionPipeline();
      
      setIsPredictionPipelineActive(false);
      setPredictions({});
      setStatus('✅ Prediction pipeline stopped');
    } catch (error) {
      setStatus(`❌ Stop pipeline error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Fetch historical data
  const fetchHistoricalData = async () => {
    if (!credentials.access_token || !credentials.api_key) {
      setStatus('❌ Please provide credentials for historical data fetch');
      return;
    }

    try {
      setLoading(true);
      setStatus('📈 Fetching historical data from Upstox...');

      const response = await liveDataAPI.fetchHistoricalData({
        access_token: credentials.access_token,
        api_key: credentials.api_key,
        instruments: selectedInstruments.map(name => popularInstruments[name] || name)
      });

      if (response.data.success) {
        setStatus(`✅ Historical data fetched: ${response.data.records_saved} records saved`);
      } else {
        setStatus(`❌ Historical fetch failed: ${response.data.message}`);
      }
    } catch (error) {
      setStatus(`❌ Historical fetch error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Clean up WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const tabs = [
    { id: 'config', name: 'Live Data Config', icon: '🔌' },
    { id: 'historical', name: 'Historical Data Fetch', icon: '📊' },
    { id: 'predictions', name: 'Live Predictions', icon: '🎯' }
  ];

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
          📡 LIVE MARKET DATA
        </h1>
        <p style={{
          fontSize: '1.2rem',
          margin: '1rem 0 0 0',
          color: 'rgba(255,255,255,0.8)',
          fontFamily: 'var(--font-primary)'
        }}>
          Real-time Upstox WebSocket Integration
        </p>
      </div>

      {/* Connection Status */}
      <Card style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h3 style={{ color: 'var(--accent-cyan)', margin: '0 0 0.5rem 0' }}>
              Connection Status
            </h3>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '1rem'
            }}>
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: isConnected ? '#51cf66' : '#ff6b6b',
                animation: isConnected ? 'pulse 2s infinite' : 'none'
              }}></div>
              <span style={{ 
                color: isConnected ? '#51cf66' : '#ff6b6b',
                fontWeight: '600'
              }}>
                {connectionStatus}
              </span>
              {isPredictionPipelineActive && (
                <span style={{
                  background: 'var(--gradient-primary)',
                  padding: '0.25rem 0.75rem',
                  borderRadius: '12px',
                  fontSize: '0.8rem',
                  fontWeight: '600'
                }}>
                  🎯 Predictions Active
                </span>
              )}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '1rem' }}>
            {!isConnected ? (
              <button
                onClick={connectToLiveData}
                disabled={loading}
                style={{
                  padding: '0.75rem 1.5rem',
                  background: loading ? 'var(--bg-secondary)' : 'var(--gradient-primary)',
                  border: 'none',
                  borderRadius: '8px',
                  color: 'white',
                  fontFamily: 'var(--font-primary)',
                  fontWeight: '600',
                  cursor: loading ? 'not-allowed' : 'pointer'
                }}
              >
                {loading ? '⏳ Connecting...' : '🔌 Connect'}
              </button>
            ) : (
              <>
                {!isPredictionPipelineActive ? (
                  <button
                    onClick={startPredictionPipeline}
                    disabled={loading}
                    style={{
                      padding: '0.75rem 1.5rem',
                      background: loading ? 'var(--bg-secondary)' : '#28a745',
                      border: 'none',
                      borderRadius: '8px',
                      color: 'white',
                      fontFamily: 'var(--font-primary)',
                      fontWeight: '600',
                      cursor: loading ? 'not-allowed' : 'pointer'
                    }}
                  >
                    {loading ? '⏳ Starting...' : '🚀 Start Predictions'}
                  </button>
                ) : (
                  <button
                    onClick={stopPredictionPipeline}
                    disabled={loading}
                    style={{
                      padding: '0.75rem 1.5rem',
                      background: loading ? 'var(--bg-secondary)' : '#dc3545',
                      border: 'none',
                      borderRadius: '8px',
                      color: 'white',
                      fontFamily: 'var(--font-primary)',
                      fontWeight: '600',
                      cursor: loading ? 'not-allowed' : 'pointer'
                    }}
                  >
                    {loading ? '⏳ Stopping...' : '🛑 Stop Predictions'}
                  </button>
                )}
                <button
                  onClick={disconnectFromLiveData}
                  disabled={loading}
                  style={{
                    padding: '0.75rem 1.5rem',
                    background: loading ? 'var(--bg-secondary)' : '#6c757d',
                    border: 'none',
                    borderRadius: '8px',
                    color: 'white',
                    fontFamily: 'var(--font-primary)',
                    fontWeight: '600',
                    cursor: loading ? 'not-allowed' : 'pointer'
                  }}
                >
                  {loading ? '⏳ Disconnecting...' : '🔌 Disconnect'}
                </button>
              </>
            )}
          </div>
        </div>
      </Card>

      {/* Configuration Tabs */}
      <Card>
        {/* Tab Navigation */}
        <div style={{
          display: 'flex',
          borderBottom: '2px solid var(--border)',
          marginBottom: '2rem',
          overflowX: 'auto'
        }}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '1rem 1.5rem',
                background: activeTab === tab.id ? 'var(--gradient-primary)' : 'transparent',
                border: 'none',
                borderBottom: activeTab === tab.id ? '3px solid var(--accent-cyan)' : '3px solid transparent',
                color: activeTab === tab.id ? 'white' : 'var(--text-primary)',
                fontFamily: 'var(--font-primary)',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                whiteSpace: 'nowrap'
              }}
            >
              {tab.icon} {tab.name}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === 'config' && (
          <div>
            <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '1.5rem' }}>
              🔌 Live Data Configuration
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Credentials */}
              <div>
                <h4 style={{ color: 'var(--accent-gold)', marginBottom: '1rem' }}>
                  📱 Upstox API Credentials
                </h4>
                <div style={{ marginBottom: '1rem' }}>
                  <label style={{
                    display: 'block',
                    color: 'var(--text-primary)',
                    marginBottom: '0.5rem',
                    fontWeight: '500'
                  }}>
                    Access Token:
                  </label>
                  <input
                    type="password"
                    value={credentials.access_token}
                    onChange={(e) => setCredentials(prev => ({
                      ...prev,
                      access_token: e.target.value
                    }))}
                    placeholder="Your Upstox access token"
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
                    API Key:
                  </label>
                  <input
                    type="password"
                    value={credentials.api_key}
                    onChange={(e) => setCredentials(prev => ({
                      ...prev,
                      api_key: e.target.value
                    }))}
                    placeholder="Your Upstox API key"
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
              </div>

              {/* Instruments */}
              <div>
                <h4 style={{ color: 'var(--accent-gold)', marginBottom: '1rem' }}>
                  📊 Instrument Configuration
                </h4>
                <div style={{ marginBottom: '1rem' }}>
                  <label style={{
                    display: 'block',
                    color: 'var(--text-primary)',
                    marginBottom: '0.5rem',
                    fontWeight: '500'
                  }}>
                    Select Instruments:
                  </label>
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                    gap: '0.5rem'
                  }}>
                    {Object.keys(popularInstruments).map(instrument => (
                      <label key={instrument} style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        color: 'var(--text-primary)',
                        cursor: 'pointer'
                      }}>
                        <input
                          type="checkbox"
                          checked={selectedInstruments.includes(instrument)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedInstruments(prev => [...prev, instrument]);
                            } else {
                              setSelectedInstruments(prev => prev.filter(i => i !== instrument));
                            }
                          }}
                          style={{ accentColor: 'var(--accent-cyan)' }}
                        />
                        {instrument}
                      </label>
                    ))}
                  </div>
                </div>
                <div>
                  <label style={{
                    display: 'block',
                    color: 'var(--text-primary)',
                    marginBottom: '0.5rem',
                    fontWeight: '500'
                  }}>
                    Custom Instrument Key:
                  </label>
                  <input
                    type="text"
                    value={customInstrument}
                    onChange={(e) => setCustomInstrument(e.target.value)}
                    placeholder="e.g., NSE_EQ|INE002A01018"
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
              </div>
            </div>
          </div>
        )}

        {activeTab === 'historical' && (
          <div>
            <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '1.5rem' }}>
              📈 Fetch Historical Data from Upstox
            </h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>
              Fetch historical 1-minute data for Nifty 50 and other instruments using Upstox API
            </p>
            
            <div style={{ textAlign: 'center' }}>
              <button
                onClick={fetchHistoricalData}
                disabled={loading || !credentials.access_token || !credentials.api_key}
                style={{
                  padding: '1rem 2rem',
                  background: loading || !credentials.access_token || !credentials.api_key
                    ? 'var(--bg-secondary)' 
                    : 'var(--gradient-primary)',
                  border: 'none',
                  borderRadius: '8px',
                  color: 'white',
                  fontFamily: 'var(--font-primary)',
                  fontWeight: '600',
                  fontSize: '1.1rem',
                  cursor: loading || !credentials.access_token || !credentials.api_key ? 'not-allowed' : 'pointer'
                }}
              >
                {loading ? '⏳ Fetching...' : '📈 Fetch Historical Data'}
              </button>
              {(!credentials.access_token || !credentials.api_key) && (
                <p style={{ color: '#ffa500', marginTop: '1rem', fontSize: '0.9rem' }}>
                  Please provide API credentials in the Live Data Config tab
                </p>
              )}
            </div>
          </div>
        )}

        {activeTab === 'predictions' && (
          <div>
            <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '1.5rem' }}>
              🎯 Live Predictions Dashboard
            </h3>
            
            {!isConnected ? (
              <div style={{
                background: 'rgba(255, 165, 0, 0.05)',
                border: '1px solid rgba(255, 165, 0, 0.2)',
                borderRadius: '8px',
                padding: '2rem',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📡</div>
                <h4 style={{ color: '#ffa500', margin: '0 0 0.5rem 0' }}>
                  Not Connected to Live Data
                </h4>
                <p style={{ color: 'var(--text-secondary)', margin: '0' }}>
                  Please connect to live data first in the Live Data Config tab
                </p>
              </div>
            ) : !isPredictionPipelineActive ? (
              <div style={{
                background: 'rgba(0, 255, 255, 0.05)',
                border: '1px solid rgba(0, 255, 255, 0.2)',
                borderRadius: '8px',
                padding: '2rem',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🎯</div>
                <h4 style={{ color: 'var(--accent-cyan)', margin: '0 0 0.5rem 0' }}>
                  Prediction Pipeline Inactive
                </h4>
                <p style={{ color: 'var(--text-secondary)', margin: '0 0 1rem 0' }}>
                  Start the prediction pipeline to see live predictions
                </p>
                <button
                  onClick={startPredictionPipeline}
                  disabled={loading}
                  style={{
                    padding: '0.75rem 1.5rem',
                    background: loading ? 'var(--bg-secondary)' : 'var(--gradient-primary)',
                    border: 'none',
                    borderRadius: '8px',
                    color: 'white',
                    fontFamily: 'var(--font-primary)',
                    fontWeight: '600',
                    cursor: loading ? 'not-allowed' : 'pointer'
                  }}
                >
                  {loading ? '⏳ Starting...' : '🚀 Start Predictions'}
                </button>
              </div>
            ) : (
              <div>
                {/* Live Market Data */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                  {Object.entries(liveData).map(([instrument, data]) => (
                    <div key={instrument} style={{
                      background: 'rgba(0, 255, 255, 0.05)',
                      border: '1px solid rgba(0, 255, 255, 0.2)',
                      borderRadius: '8px',
                      padding: '1rem'
                    }}>
                      <h5 style={{ color: 'var(--accent-cyan)', margin: '0 0 0.5rem 0', fontSize: '0.9rem' }}>
                        {Object.keys(popularInstruments).find(key => popularInstruments[key] === instrument) || instrument}
                      </h5>
                      <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--accent-gold)', marginBottom: '0.25rem' }}>
                        ₹{data.ltp?.toFixed(2) || 'N/A'}
                      </div>
                      <div style={{
                        fontSize: '0.8rem',
                        color: (data.change || 0) >= 0 ? '#51cf66' : '#ff6b6b'
                      }}>
                        {(data.change || 0) >= 0 ? '▲' : '▼'} {Math.abs(data.change || 0).toFixed(2)} ({((data.change || 0) / (data.ltp || 1) * 100).toFixed(2)}%)
                      </div>
                    </div>
                  ))}
                </div>

                {/* Live Predictions */}
                {Object.keys(predictions).length > 0 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(predictions).map(([instrument, prediction]) => (
                      <div key={instrument} style={{
                        background: 'rgba(255, 0, 128, 0.05)',
                        border: '1px solid rgba(255, 0, 128, 0.2)',
                        borderRadius: '8px',
                        padding: '1rem'
                      }}>
                        <h5 style={{ color: '#ff0080', margin: '0 0 1rem 0' }}>
                          🎯 {Object.keys(popularInstruments).find(key => popularInstruments[key] === instrument) || instrument}
                        </h5>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))', gap: '0.5rem' }}>
                          <div style={{ textAlign: 'center' }}>
                            <div style={{ color: '#ff0080', fontSize: '1.2rem', fontWeight: '600' }}>
                              {prediction.direction || 'N/A'}
                            </div>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                              Direction
                            </div>
                          </div>
                          <div style={{ textAlign: 'center' }}>
                            <div style={{ color: '#ff0080', fontSize: '1.2rem', fontWeight: '600' }}>
                              {(prediction.confidence * 100)?.toFixed(1) || 'N/A'}%
                            </div>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                              Confidence
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Status Message */}
      {status && (
        <div style={{
          position: 'fixed',
          bottom: '2rem',
          right: '2rem',
          padding: '1rem 1.5rem',
          background: status.includes('❌') 
            ? 'rgba(255, 0, 0, 0.9)' 
            : status.includes('✅')
            ? 'rgba(0, 255, 0, 0.9)'
            : 'rgba(0, 255, 255, 0.9)',
          border: `1px solid ${
            status.includes('❌') ? '#ff0000' : 
            status.includes('✅') ? '#00ff00' : '#00ffff'
          }`,
          borderRadius: '8px',
          color: 'white',
          fontFamily: 'var(--font-primary)',
          fontWeight: '600',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
          zIndex: 1000,
          maxWidth: '400px'
        }}>
          {status}
        </div>
      )}
    </div>
  );
};

export default LiveData;